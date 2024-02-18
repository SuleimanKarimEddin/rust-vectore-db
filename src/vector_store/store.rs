use super::request_util;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use wasm_bindgen::prelude::*;

/// `VectorStore` is a data structure that allows storing and querying
/// high-dimensional vectors (embeddings) efficiently using a k-d tree.
///
/// # Fields
///
/// * `db`: A `KdTree` data structure for indexing the high-dimensional vectors.
/// * `url`: An optional field to store a URL related to the `VectorStore`.
/// * `words`: A list of words, where each word is associated with a vector in the `db`.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct VectorStore {
    db: KdTree<f64, i32, Vec<f64>>,
    url: String,
    words: Vec<String>,
}

#[wasm_bindgen]
impl VectorStore {
    /// Constructs a new instance of the struct, initializing the URL, vector dimension, and creating an empty KdTree.
    /// If the vector dimension is not provided, it defaults to 384.
    ///
    /// # Arguments
    ///
    /// * `url` - An optional String representing the URL.
    /// * `vectore_dimension` - An optional usize representing the dimension of vectors. If not provided, defaults to 384.
    ///
    /// # Returns
    ///
    /// A new instance of the struct with the specified URL, vector dimension, and an empty KdTree.
    pub fn new(url: Option<String>, vectore_dimension: Option<usize>) -> Self {
        let vectore_dimension = vectore_dimension.unwrap_or(384);
        let url = url.unwrap_or("https://embidded-serever.onrender.com/".to_string());
        Self {
            url,
            db: KdTree::new(vectore_dimension),
            words: Vec::new(),
        }
    }
    /// Asynchronously adds a vector by word to the KdTree.
    ///
    /// # Arguments
    ///
    /// * `name` - A String representing the word.
    ///
    /// # Returns
    ///
    /// * `Result<(), JsValue>` - A Result indicating success or an error wrapped in JsValue.
    ///                          - Ok(()) if the operation succeeds.
    ///                          - Err(JsValue) if there is an error during the asynchronous operation.
    ///
    /// # Remarks
    ///
    /// This function retrieves the embedding vector for the given word asynchronously using `request_util::get_embbide_request`.
    /// If successful, it adds the word and its vector to the KdTree.

    pub async fn add_vectore_by_word(&mut self, name: String) -> Result<(), JsValue> {
        let vector = request_util::get_embbide_request(&name, &self.url).await?;
        self.add(name, vector.0);
        Ok(())
    }
    /// Asynchronously adds vectors by text to the KdTree.
    ///
    /// # Arguments
    ///
    /// * `text` - A String representing the text containing words separated by newline characters.
    ///
    /// # Returns
    ///
    /// * `Result<(), JsValue>` - A Result indicating success or an error wrapped in JsValue.
    ///                          - Ok(()) if the operation succeeds.
    ///                          - Err(JsValue) if there is an error during the asynchronous operation.
    ///
    /// # Remarks
    ///
    /// This function splits the input text into an array of words using newline ('\n') as the delimiter.
    /// It then iterates over chunks of 50 words at a time, retrieves their embedding vectors asynchronously
    /// using `request_util::get_embbide_request_array`, and adds each word and its corresponding vector to the KdTree.

    pub async fn add_vectore_by_text(&mut self, text: String) -> Result<(), JsValue> {
        let word_array = text.split('\n').collect::<Vec<&str>>();
        for i in word_array.chunks(20) {
            let vector = request_util::get_embbide_request_array(i, &self.url).await?;
            for j in 0..vector.0.len() {
                self.add(i[j].to_string(), vector.0[j].clone());
            }
        }

        Ok(())
    }
    /// Retrieves a clone of the words stored in the KdTree.
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - A vector containing the words stored in the KdTree.

    pub fn get_words(&self) -> Vec<String> {
        self.words.clone()
    }
    /// Asynchronously retrieves a collection of similar words to the given word from the KdTree.
    ///
    /// # Arguments
    ///
    /// * `word` - A String representing the word to find similar words for.
    /// * `top_k` - A usize specifying the number of similar words to retrieve.
    ///
    /// # Returns
    ///
    /// * `Result<StringCollection, JsValue>` - A Result indicating success or an error wrapped in JsValue.
    ///                                        - Ok(StringCollection) containing the collection of similar words.
    ///                                        - Err(JsValue) if there is an error during the asynchronous operation.
    ///
    /// # Remarks
    ///
    /// This function retrieves the embedding vector for the given word asynchronously using `request_util::get_embbide_request`.
    /// It then finds the top `top_k` similar words based on the cosine similarity of their vectors in the KdTree.
    /// The result is wrapped in a StringCollection.
    pub async fn similar_words(
        &self,
        word: String,
        top_k: usize,
    ) -> Result<StringCollection, JsValue> {
        let vector = request_util::get_embbide_request(&word, &self.url).await?;
        let top_k_vec = self.find_similers(&vector.0, top_k);
        Ok(StringCollection(top_k_vec))
    }
}

impl VectorStore {
    pub fn add(&mut self, word: String, vector: Vec<f64>) {
        self.words.push(word);
        self.db.add(vector, self.words.len() as i32 - 1).unwrap();
    }

    pub fn find_similers(&self, vector: &Vec<f64>, top_k: usize) -> Vec<String> {
        let result = self.db.nearest(vector, top_k, &squared_euclidean).unwrap();
        result
            .iter()
            .map(|&(_, i)| self.words[*i as usize].clone())
            .collect::<Vec<String>>()
    }
}

#[wasm_bindgen]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct F64Collection(Vec<f64>);

#[wasm_bindgen]
impl F64Collection {
    /// Constructs a new F64Collection from a vector of f64 values.
    ///
    /// # Arguments
    ///
    /// * `vector` - A vector containing f64 values.
    ///
    /// # Returns
    ///
    /// A new F64Collection instance containing the provided f64 values.
    pub fn new(vector: Vec<f64>) -> Self {
        Self(vector)
    }
}

#[wasm_bindgen]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct F64ArrayCollection(Vec<Vec<f64>>);

impl F64ArrayCollection {
    /// Constructs a new F64ArrayCollection from a vector of vectors of f64 values.
    ///
    /// # Arguments
    ///
    /// * `vector` - A vector containing vectors of f64 values.
    ///
    /// # Returns
    ///
    /// A new F64ArrayCollection instance containing the provided vectors of f64 values.
    pub fn new(vector: Vec<Vec<f64>>) -> Self {
        Self(vector)
    }

    /// Retrieves the first vector of f64 values from the collection.
    ///
    /// # Returns
    ///
    /// The first vector of f64 values in the collection.
    pub fn get_first_vector(&self) -> Vec<f64> {
        self.0.get(0).unwrap().clone()
    }

    /// Returns the number of vectors in the collection.
    ///
    /// # Returns
    ///
    /// The number of vectors in the collection.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Retrieves a vector of f64 values from the collection by its index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the vector to retrieve.
    ///
    /// # Returns
    ///
    /// The vector of f64 values at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get_vector_by_index(&self, index: usize) -> Vec<f64> {
        self.0.get(index).unwrap().clone()
    }
}

/// A collection of strings.
/// you can retrevie the strings from the collection using `get_vector` method
///  #example
///  let collection = StringCollection::new(vec!["hello".to_string(), "world".to_string()]);
///  let vector = collection.get_vector();
#[wasm_bindgen]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StringCollection(Vec<String>);

#[wasm_bindgen]
impl StringCollection {
    /// Constructs a new StringCollection from a vector of strings.
    ///
    /// # Arguments
    ///
    /// * `vector` - A vector containing strings.
    ///
    /// # Returns
    ///
    /// A new StringCollection instance containing the provided strings.
    pub fn new(vector: Vec<String>) -> Self {
        Self(vector)
    }

    /// Returns the number of strings in the collection.
    ///
    /// # Returns
    ///
    /// The number of strings in the collection.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Retrieves all strings from the collection.
    ///
    /// # Returns
    ///
    /// A vector containing all strings in the collection.
    pub fn get_vector(&self) -> Vec<String> {
        self.0.clone()
    }
}
