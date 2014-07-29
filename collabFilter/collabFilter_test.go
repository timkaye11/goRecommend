package collabFilter

import (
	"fmt"
	"testing"
)

func Assert(t *testing.T, condition bool, args ...interface{}) {
	if !condition {
		t.Fatal(args)
	}
}

func TestSortMap(t *testing.T) {
	to_sort := make(map[float64]string, 0)
	to_sort[0.23423] = "Drew"
	to_sort[0.99999] = "Tim"
	to_sort[0.44444] = "Kyle"
	names, scores := sortMap(to_sort)

	Assert(t, names[0] == "Tim", scores[0] == 0.99999)
}

func TestCosineSim(t *testing.T) {
	x := []float64{2, 3, 4, 1}
	y := []float64{3, 0, 3, 4}
	cosine_sim := CosineSim(x, y)
	// should be like 0.688.... Checked using R
	Assert(t, cosine_sim > 0.687, cosine_sim < 0.689)
}

func TestJaccard(t *testing.T) {
	x := []float64{1, 1, 0, 1, 1}
	y := []float64{1, 1, 1, 0, 1}
	jaccard_sim := Jaccard(x, y)
	// should be 0.6
	Assert(t, jaccard_sim == 0.6)
}

func TestRatingRecommendations(t *testing.T) {
	// User product matrix. 0 or math.NaN indicates products not viewed by user.
	// Uses cosine similarity.
	prefs := MakeRatingMatrix([]float64{
		2, 3, 4, 1, 5,
		3, 0, 3, 3, 0,
		4, 4, 1, 2, 3,
		2, 4, 0, 3, 4,
		3, 1, 3, 0, 4}, 5, 5)
	products := []string{"Spiderman", "Big Momma's House", "Vanilla Sky", "Pacific Rim", "The Mask"}
	// gets recommendations for user 1 (second row) for un-rated products.
	prods, scores, err := GetRecommendations(prefs, 1, products)
	fmt.Println(prods)

	// make sure these recommendations make sense, and match up to python implementation.
	Assert(t, err == nil)
	Assert(t, prods[0] == "The Mask", prods[1] == "Big Momma's House")
	Assert(t, scores[0] > 3.80)
}

func TestBinaryRecommendations(t *testing.T) {
	// For a binary matrix, use the getBinaryRecommendations function in the exact same way.
	// Uses Jaccard Similarity to return confidence/probabality of user's recommendations
	binaryPrefs := MakeRatingMatrix([]float64{
		1, 1, 1, 1, 0,
		0, 1, 1, 0, 1,
		1, 1, 1, 1, 1,
		1, 1, 0, 0, 1,
		1, 0, 1, 1, 1}, 5, 5)
	products := []string{"Spiderman", "Big Momma's House", "Vanilla Sky", "Pacific Rim", "The Mask"}
	// Returns recommended products for User ID 1 (second row) in descending order, w/ corresponding confidence/probability,
	// and error - if applicable.
	prods, scores, err := GetBinaryRecommendations(binaryPrefs, 1, products)

	Assert(t, err == nil)
	Assert(t, prods[0] == "Spiderman", prods[1] == "Pacific Rim")
	Assert(t, scores[0] > 0.4, scores[1] < 0.3)

}
