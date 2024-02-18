use crate::vector_store::store::F64ArrayCollection;

use super::store::F64Collection;
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

#[derive(Deserialize, Debug)]
pub struct MessageDeserializer {
    pub data: Vec<f64>,
}

struct RequestSinglton {
    pub request_init: Option<RequestInit>,
}
impl RequestSinglton {
    pub fn get(&mut self) -> &RequestInit {
        self.request_init.get_or_insert_with(|| {
            let mut opts = RequestInit::new();
            opts.method("GET");
            opts.mode(RequestMode::Cors);
            opts
        })
    }
    pub fn new() -> Self {
        Self { request_init: None }
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

pub async fn get_embbide_request(word: &str, url: &str) -> Result<F64Collection, JsValue> {
    let mut opts = RequestSinglton::new();
    let url = format!("{}?word={}", url, word);
    log(&url);
    let request = Request::new_with_str_and_init(&url, opts.get())?;
    request.headers().set("Accept", "application/json")?;
    let window = web_sys::window().unwrap_throw();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into().unwrap();
    let json = JsFuture::from(resp.json()?).await?;
    let result: MessageDeserializer = serde_wasm_bindgen::from_value(json)?;

    let vectore = F64Collection::new(result.data);
    Ok(vectore)
}
#[derive(Deserialize, Debug)]
pub struct ArrayMessageDeserializer {
    pub data: Vec<Vec<f64>>,
}

pub async fn get_embbide_request_array(
    word_array: &[&str],
    url: &str,
) -> Result<F64ArrayCollection, JsValue> {
    log(&url);
    let mut opts = RequestInit::new();
    opts.method("POST");
    opts.mode(RequestMode::Cors);
    let json_data = serde_json::to_string(word_array).unwrap();
    let body = JsValue::from_str(&json_data);
    opts.body(Some(&body));
    let url = format!("{}list", url);
    let request = Request::new_with_str_and_init(&url, &opts)?;
    request.headers().set("Accept", "application/json")?;
    request.headers().set("Content-Type", "application/json")?;
    let window = web_sys::window().unwrap_throw();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into().unwrap();
    let json = JsFuture::from(resp.json()?).await?;
    let result: ArrayMessageDeserializer = serde_wasm_bindgen::from_value(json)?;
    let vectore = F64ArrayCollection::new(result.data);
    Ok(vectore)
}
