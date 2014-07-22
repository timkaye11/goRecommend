package ALS

import (
	"fmt"
	. "github.com/skelterjohn/go.matrix"
	"testing"
)

func Assert(t *testing.T, condition bool, args ...interface{}) {
	if !condition {
		t.Fatal(args)
	}
}

func TestMatrices(t *testing.T) {
	Q := MakeDenseMatrix([]float64{5, 5, 5, 0, 1,
		0, 0, 0, 4, 1,
		2, 0, 4, 1, 0,
		5, 2, 0, 1, 0}, 4, 5)
	sum := sumMatrix(MakeWeightMatrix(Q))
	Assert(t, sum == 12)

	newRow := setRow(Q, 1, []float64{1, 1, 3, 1, 2})
	Assert(t, newRow.Get(1, 2) == 3)

}

func TestExplicit(t *testing.T) {
	Q := MakeDenseMatrix([]float64{5, 5, 5, 0, 1,
		0, 0, 0, 4, 1,
		1, 2, 3, 3, 1,
		2, 0, 4, 1, 0,
		5, 2, 0, 1, 0}, 5, 5)

	n_factors := 5
	n_iterations := 10
	lambda := 0.01

	Qhat, err := TrainALS(Q, n_factors, n_iterations, lambda)
	if Qhat.Rows() != Q.Rows() || Qhat.Cols() != Q.Cols() {
		t.Errorf("Unexpected Dimensions. Got %v & %v", Qhat.Rows(), Qhat.Cols())
	}
	Assert(t, Qhat.Get(0, 3) > 2)
	Assert(t, err < 1)
}

func TestImplicit(t *testing.T) {
	Q := MakeDenseMatrix([]float64{10, 3, 2,
		0, 2, 0,
		5, 1, 0}, 3, 3)
	n_factors := 3
	n_iterations := 5
	lambda := 0.01

	Qhat := TrainALS_Implicit(Q, n_factors, n_iterations, lambda)
	fmt.Println(Qhat)
	Assert(t, Qhat.Get(1, 0) > 0)
}

func TestPredictions(t *testing.T) {
	Q := MakeDenseMatrix([]float64{5, 5, 5, 0, 1,
		0, 0, 0, 4, 1,
		1, 2, 3, 3, 1,
		2, 0, 4, 1, 0,
		5, 2, 0, 1, 0}, 5, 5)

	Qhat, _ := TrainALS(Q, 5, 10, 0.01)
	fmt.Printf("Prediction Test, Prediction Matrix: %v", Qhat)
	// If Product Names is nil, then returns top indices for each user. Returns in descending order.
	products := []string{"Macy Gray", "The Black Keys", "Spoon", "A Tribe Called Quest", "Kanye West"}
	preds, _ := GetTopNRecommendations(Q, Qhat, 1, 2, products)
	fmt.Println(preds)
	Assert(t, preds[0] == "Spoon")
}
