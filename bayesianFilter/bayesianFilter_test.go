package bayesianFilter

import (
	. "github.com/skelterjohn/go.matrix"
	. "goCF/bayesianFilter"
	"testing"
)

func TestBayesian(t *testing.T) {

	matrix := MakeDenseMatrix([]float64{4, NA, 5, 5,
		4, 2, 1, NA,
		3, NA, 2, 4,
		4, 4, NA, NA,
		2, 1, 3, 5}, 5, 4)

	predictedValue := 4

	_, pred, err := BayesianFilter(matrix, 0, 1)
	if pred != predictedValue {
		t.Errorf("Predicted value %v should be %v", pred, predictedValue)
		t.Fatal()
	}
	if err != nil {
		t.Errorf("Error in BayesianFilter: %v", err)
		t.Fatal()
	}
}
