// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_Normalize() implementation using the geos crate
pub fn st_normalize_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STNormalize {})
}

#[derive(Debug)]
struct STNormalize {}

impl SedonaScalarKernel for STNormalize {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            args[0].clone(),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(executor.num_iterations(), executor.num_iterations() * 128);
        
        executor.execute_wkb_void(|maybe_geom| {
            match maybe_geom {
                Some(mut geos_geom) => {
                    // Perform the normalization
                    geos_geom.normalize().map_err(|e| {
                        DataFusionError::Execution(format!("GEOS Normalize Error: {e}"))
                    })?;
                    
                    // Convert GEOS geometry back to WKB bytes for the Arrow array
                    let wkb_bytes = geos_geom.to_wkb().map_err(|e| {
                        DataFusionError::Execution(format!("WKB Serialization Error: {e}"))
                    })?;
                    
                    builder.append_value(wkb_bytes);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, GenericBinaryArray};
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_normalize", st_normalize_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        tester.assert_return_type(DataType::Binary); 

        let input_wkt = "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))";
        let expected_wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"; 
        
        let result = tester.invoke_scalar(input_wkt).unwrap();

        tester.assert_scalar_result_equals(result, expected_wkt);

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let batch_input = vec![
            Some("POINT(2 1)"), 
            None,
            Some("POLYGON((1 1, 1 0, 0 0, 0 1, 1 1))"),
        ];
        
        let batch_result = tester.invoke_wkb_array(batch_input).unwrap();
        assert_eq!(batch_result.len(), 3);
        assert!(batch_result.is_null(1));
    }
}