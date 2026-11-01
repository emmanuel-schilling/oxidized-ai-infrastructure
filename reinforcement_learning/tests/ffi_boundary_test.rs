// reinforcement_learning/tests/ffi_boundary_test.rs

mod common;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::thread;

// =========================================================================
// CATEGORY 1: Environment & Initialization Test
// =========================================================================
#[test]
fn test_embedded_gymnasium_initialization() {
    common::ensure_python_initialized();

    Python::with_gil(|py| {
        let gym = py.import("gymnasium");
        assert!(
            gym.is_ok(),
            "FAIL: Local environment is missing Gymnasium package!"
        );

        let env = gym.unwrap().call_method1("make", ("CartPole-v1",));
        assert!(
            env.is_ok(),
            "FAIL: Could not instantiate CartPole over the local FFI boundary!"
        );
    });
}

// =========================================================================
// CATEGORY 2: NumPy Array Precision & Memory Layout
// =========================================================================
#[test]
fn test_numpy_array_boundary_and_precision_casting() {
    common::ensure_python_initialized();

    Python::with_gil(|py| {
        let np = py.import("numpy").expect("FAIL: NumPy package missing");

        // Gymnasium observations are typically float32; verify safe casting to Rust f64
        let float32_data: Vec<f32> = vec![0.5, -1.25, 3.125, 0.0625];
        let py_array = np
            .call_method1("array", (float32_data.clone(),))
            .expect("FAIL: Could not construct NumPy array");

        let shape: Vec<usize> = py_array.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![4], "FAIL: NumPy shape mismatch across boundary");

        // Validate that PyO3 handles the f32 -> f64 upcast seamlessly
        let extracted_f64: Vec<f64> = py_array.call_method0("tolist").unwrap().extract().unwrap();
        for (orig, upcast) in float32_data.iter().zip(extracted_f64.iter()) {
            assert!(
                ((*orig as f64) - *upcast).abs() < 1e-6,
                "FAIL: Precision loss on FFI upcast"
            );
        }
    });
}

// =========================================================================
// CATEGORY 3: State Management & Multi-Threaded GIL Safety
// =========================================================================
#[test]
fn test_multi_threaded_gil_contention() {
    common::ensure_python_initialized();

    // Spawn multiple native OS threads competing for the GIL
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            thread::spawn(move || {
                Python::with_gil(|py| {
                    let math = py
                        .import("math")
                        .expect("FAIL: Could not import math module");
                    let res: f64 = math
                        .call_method1("sqrt", (thread_id as f64 + 1.0,))
                        .unwrap()
                        .extract()
                        .unwrap();
                    assert!(res > 0.0);
                });
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("FAIL: Worker thread panicked during GIL contention");
    }
}

// =========================================================================
// CATEGORY 4: Exception Propagation Across FFI Boundary
// =========================================================================
#[test]
fn test_python_exception_propagation_to_pyerr() {
    common::ensure_python_initialized();

    Python::with_gil(|py| {
        let gym = py.import("gymnasium").unwrap();

        // Intentionally request an invalid environment ID
        let bad_env_result = gym.call_method1("make", ("NonExistentGymEnv-v999",));

        assert!(
            bad_env_result.is_err(),
            "FAIL: Calling gymnasium.make() with an invalid environment should return a PyErr"
        );

        let py_err = bad_env_result.unwrap_err();
        let type_name_pystr = py_err.get_type(py).name().unwrap();
        let err_name = type_name_pystr.to_str().unwrap();

        // Gymnasium raises `NameNotFound` when an environment ID doesn't exist
        assert!(
            err_name.contains("NameNotFound") || err_name.contains("Error"),
            "FAIL: Expected Gymnasium error, got: {:?}",
            py_err
        );
    });
}

// =========================================================================
// CATEGORY 5: Gymnasium Lifecycle Protocol
// =========================================================================
#[test]
fn test_gymnasium_lifecycle_flow() {
    common::ensure_python_initialized();

    Python::with_gil(|py| {
        let gym = py.import("gymnasium").unwrap();
        let env = gym.call_method1("make", ("CartPole-v1",)).unwrap();

        // 1. Reset
        let reset_res = env.call_method0("reset").unwrap();
        let reset_tuple = reset_res.downcast_into::<PyTuple>().unwrap();
        let obs_init: Vec<f64> = reset_tuple.get_item(0).unwrap().extract().unwrap();
        let info_init = reset_tuple.get_item(1).unwrap().downcast_into::<PyDict>();

        assert_eq!(obs_init.len(), 4);
        assert!(info_init.is_ok());

        // 2. Step
        let step_res = env.call_method1("step", (1,)).unwrap();
        let step_tuple = step_res.downcast_into::<PyTuple>().unwrap();
        assert_eq!(step_tuple.len(), 5);

        let reward: f64 = step_tuple.get_item(1).unwrap().extract().unwrap();
        let terminated: bool = step_tuple.get_item(2).unwrap().extract().unwrap();
        let truncated: bool = step_tuple.get_item(3).unwrap().extract().unwrap();

        assert!(reward > 0.0);
        assert!(!terminated);
        assert!(!truncated);
    });
}
