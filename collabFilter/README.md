### Memory-Based Collaborative Filtering (in Go)

To quote wikipedia, "this is used for making recommendations". 

This uses a neighest-neighbor based algorithm to compare a users product selection with other users and recommends products for a selected uesr. Cosine similarity is used when there is an explicit rating system, and jaccard similarity is used for the binary case. 

---
To use,  download the package:
``` go get github.com/timkaye11/goRecommend/collabFilter```

---
#### Example

```
import "github.com/timkaye11/goRecommend/collabFilter"

func main() {
	// User product matrix. 0 or math.NaN indicates products not viewed by user.
	// Uses cosine similarity.


	// arguments to **MakeRatingMatrix** are: data, nrows, ncols. 
	prefs := MakeRatingMatrix([]float64{
		2, 3, 4, 1, 5,
		3, 0, 3, 3, 0,
		4, 4, 1, 2, 3,
		2, 4, 0, 3, 4,
		3, 1, 3, 0, 4}, 5, 5)

	// product titles <- column titles for prefs matrix
	products := []string{"Spiderman", "Big Momma's House", "Vanilla Sky", "Pacific Rim", "The Mask"}
	// gets recommendations for user 1 (second row) for un-rated products.
	prods, scores, err := getRecommendations(prefs, 1, products)
	if err != nil {
		fmt.Println("WHAT!?")
	}
	fmt.Printf("\nRecommended Products are: %v, with scores: %v", prods, scores)

	// For a binary matrix, use the getBinaryRecommendations function in the exact same way.
	// Uses Jaccard Similarity to return confidence/probabality of user's recommendations
	binaryPrefs := MakeRatingMatrix([]float64{
		1, 1, 1, 1, 0,
		0, 1, 1, 0, 1,
		1, 1, 1, 1, 1,
		1, 1, 0, 0, 1,
		1, 0, 1, 1, 1}, 5, 5)
	// Returns recommended products for User ID 1 (second row) in descending order, w/ corresponding confidence/probability,
	// and error - if applicable.
	prods, scores, _ := getBinaryRecommendations(binaryPrefs, 1, products)
	...


}
```


