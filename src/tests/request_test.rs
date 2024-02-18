#![cfg(test)]

use std::vec;

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

use crate::vector_store::store::VectorStore;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_array(s: Vec<String>);
}
#[wasm_bindgen_test]
async fn test_add_vectore_by_text() {
    let mut vectore_db = VectorStore::new(None, None);
    let text = "car\nred\nbuss\nargo";
    let _ = vectore_db.add_vectore_by_text(text.to_string()).await;

    let gess = vectore_db
        .similar_words("color".to_string(), 2)
        .await
        .unwrap();

    assert_eq!(
        gess.get_vector(),
        vec!["red".to_string(), "car".to_string()]
    );
}

#[wasm_bindgen_test]
async fn test_add_vectore() {
    let mut vectore_db = VectorStore::new(None, None);
    let text = "car";
    let _ = vectore_db.add_vectore_by_word(text.to_string()).await;

    let gess = vectore_db
        .similar_words("color".to_string(), 1)
        .await
        .unwrap();

    assert_eq!(gess.get_vector(), vec!["car".to_string()]);
}
