# What's "Cosine Similarity" anyways?

<sup>*"If you have less than 1000 entries, you may not need a vector DB at all"*</sup>

<ins>**Nerd answer**</ins>: It's the normalized dot product of two embedding vectors. 

<ins>**Executive summary**</ins>: it's a computed "inverse distance" between two pieces of data. 1 would mean two data points are identical, 0 would mean they have nothing in common.

What does that mean? 

When you extract an embedding from your text, image, sound, smell, thought, or whatever, you get a vector. 

e.g.:

```javascript
const embedding = await embeddings.create("The quick brown fox jumped over the lazy dog") // simplified
```
You'll get a vector as a response:
```javascript
"embedding": [
  0.0023064255,
  -0.009327292,
  //.... (1536 floats total for ada-002)
  -0.0028842222,
],
```
<sup>(source: https://platform.openai.com/docs/api-reference/embeddings/create)</sup>

> <ins>vector</ins>: you can almost think of this as a coordinate. If you ever played the game [Battleship](https://en.wikipedia.org/wiki/Battleship_(game)), \[B,6\] ("B-6") could be a vector that you shot at. Here, instead of two, we use thousands of dimensions. And instead of A to J and 1 to 10, we use the range -1 to 1 in every dimension.

In the case of OpenAI, these vectors will already be normalized for you.

> <ins>normalized</ins> means that the length (or magnitude) of the vector (the hypotenuse) is 1. We use the pythagorean theorem: $\sqrt{a^2 + b^2} = c$, and $c = 1$. For a 1536 dimensional vector, we just do $\sqrt{a^2 + b^2 + c^2 + d^2 + e^2 + \text{... (1536 terms in total)}} = 1$.

But here's the function for reference:

```javascript
function normalize(vector) {
  // first, we get the magnitude
  const magnitude_squared = vector.map(element => element*element) // square each element
    .reduce((a,b) => a+b) // and then add them all up
  const magnitude = Math.sqrt(magnitude_squared) // and finally take the square root.

  // then we normalize.
  return vector.map(element => element/magnitude) // simply by dividing each element by the magnitude.
}
```
> notice: normalizing a normalized vector won't change anything: the magnitude will be 1, and dividing by 1 doesn't do anything, other than waste CPU cycles.

Once they are normalized, we compute the dot product $V_1 \cdot V_2$ between two vectors.

```javascript
function dot_product(v1, v2) {
    return v1.map((_, index) => v1[index] * v2[index]) // for each element in the vector, multiply with same index element in other vector,
      .reduce((a,b) => a+b) // and then just sum everything up.
}
```

And you have your cosine similarity. That's it. 

## How do I use this?

Generally, you have a <ins>query</ins>, and a <ins>corpus</ins>. 

Your <ins>query</ins> is what you're actively looking for, e.g.: 

```javascript
const query = normalize(await embeddings.create("The quick brown fox")) // normalization may be optional with some models
```

And your <ins>corpus</ins> stores the stuff you want to look through, containing pre-computed embeddings:

```javascript
const corpus = [
  {
    id: 1,
    text: "Why did the chicken cross the road?",
    vector: [0.0028064255, -0.009327292, /*...*/]
  },
  {
    id: 2,
    text: "The quick brown fox jumped over the lazy dog",
    vector: [0.0023064255, -0.009327292, /*...*/]
  },
  {
    id: 3,
    text: "She sells seashells by the seashore.",
    vector: [0.0065063255, -0.009456622, /*...*/]
  },
  //...
]
```

Then, you can just go through that elementwise:

```javascript
const results = corpus
  .map(element => {
    return { 
      "cosim": dot_product(element.vector, query), // compute the cosine similarity
      "data": element // we pass a reference to the element in the corpus, because we want to sort this later
    }
  }))
  .sort( (a, b) => b.cosim - a.cosim ) // sort in descending order
```

And if we did everything right, the closest match should now be at the front of the list

```javascript
console.log(results[0].data)
```
output:
```javascript
{
  id: 2,
  text: "The quick brown fox jumped over the lazy dog",
  vector: [0.0023064255, -0.009327292, /*...*/]
}
```

Congratulations! You've just implemented the Pinecone/Milvus/Weaviate/Chroma/Faiss/etc, FLAT index. 

And you can even run it directly in your front end!
<ins>⚠️*Warning! Do not call OpenAI/Anthropic/Huggingface/Voyage/Vertex/etc APIs directly from your front end. You will need a proxy - otherwise your API keys WILL get stolen!</ins>

## Operational advice:

If you have less than 1000 elements per corpus, you might be able to get away with simply *not* using a vector database. Just a naive for loop, iterating over each element might be super good enough for your MVP.

Some embedding models have more dimensions, some have fewer. More dimensions, more compute cost (although sometimes also more accuracy). 

Some embedding models support matryoshka embeddings - this means that the most important dimensions are at the front of the vector, meaning that you can truncate the vector (i.e., use half the dimensions, allowing you to double the lookup speed)
OpenAI offers something similar with their dimensions parameter: https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions. An [experiment by one of the OpenAI community members](https://community.openai.com/t/it-looks-like-text-embedding-3-embeddings-are-truncated-scaled-versions-from-higher-dim-version/602276/14) seems to indicate that there's an exponential relationship between dimensions and accurancy:- If your data is diverse enough, 500-1000 dimensions might be good enough for your purposes.

---

Thanks for reading! Questions? Comments? Feel free to [open an issue here](https://github.com/neurofleet/blog/issues/new) or visit the OpenAI community forums [https://community.openai.com/](https://community.openai.com/) :)
