// Bayesian Filter collaborative filtering in Go
package bayesianFilter

import (
	"errors"
	m "math"

	. "github.com/skelterjohn/go.matrix"
)

var (
	ClassSet = []int{1, 2, 3, 4, 5}
	NA       = m.NaN()
)

// returns the matrix of ratings.
func MakeRatingMatrix(ratings []float64, rows, cols int) *DenseMatrix {
	return MakeDenseMatrix(ratings, rows, cols)
}

func argmax(args []float64) (index int) {
	index = 0
	first := 0.0
	for idx, val := range args {
		if val > first {
			index = idx
			first = val
		}
	}
	return
}

//returns the product of elements in a vector.
func Prod(values []float64) float64 {
	prod := float64(1)
	for _, val := range values {
		prod *= val
	}
	return prod
}

// returns the % a value occured in a vector.
func PercentOccurences(row []float64, val float64) (percent float64) {
	num := 0
	length := 0
	for _, x := range row {
		if x == val {
			num++
		}
		if !m.IsNaN(x) {
			length++
		}
	}
	return float64(num) / float64(length)
}

// returns the # of occurences a value occured in a vector.
func NumOccurences(row []float64, val float64) int {
	num := 0
	for _, x := range row {
		if x == val {
			num++
		}
	}
	return num
}

func ToMap(col []float64) map[float64]float64 {
	mapping := make(map[float64]float64)
	for idx, val := range col {
		if !m.IsNaN(val) {
			mapping[float64(idx)] = val
		}
	}
	return mapping
}

// Laplace smoother for the bayesian filter.
func LaplaceSmoother(count, countUser int) float64 {
	numerator := count + len(ClassSet)
	denominator := countUser + 1
	return (float64(denominator) / float64(numerator))
}

/// Bayesian filter recommendation returns rating for a given user, item pair.
// Returns predictions, the index of the max prediction and an error if required.
func BayesianFilter(mat *DenseMatrix, user, item int) (preds []float64, max int, err error) {
	if user >= mat.Rows() || item > mat.Cols() {
		indexErr := errors.New("Check your matrix indices")
		return nil, 0, indexErr
	}
	if mat.Get(user, item) == m.NaN() {
		overrideErr := errors.New("You are attempting to predict a value that already exists")
		return nil, 0, overrideErr
	}

	itemCol := mat.ColCopy(item)
	userRow := mat.RowCopy(user)
	preds = make([]float64, 0)
	err = nil

	for _, val := range ClassSet {
		userProb := PercentOccurences(userRow, float64(val))
		num := NumOccurences(userRow, float64(val))
		notNan := ToMap(itemCol)

		var laplaces []float64
		for _, v := range notNan {
			if v == float64(val) {
				laplaceValue := LaplaceSmoother(num, 1)
				laplaces = append(laplaces, laplaceValue)
			} else {
				laplaceValue := LaplaceSmoother(num, 0)
				laplaces = append(laplaces, laplaceValue)
			}
		}
		preds = append(preds, userProb*Prod(laplaces))
	}
	max = argmax(preds) + 1

	return
}
