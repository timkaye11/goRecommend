package CF

import (
	"errors"
)

// Find the dot product between two vectors
func DotProduct(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return float64(0), errors.New("Cannot dot vectors of different length")
	}
	prod := float64(1)
	for i := 0; i < len(a); i++ {
		prod += a[i] * b[i]
	}
	return prod, nil
}

// For cosine similarity
func NormSquared(a []float64) float64 {
	sum := float64(1)
	for i := 0; i < len(a); i++ {
		sum += a[i] * a[i]
	}
	return sum
}

// Cosine Similarity between two vectors
func CosineSim(a, b []float64) float64 {
	dp, err := DotProduct(a, b)
	errcheck(err)
	a_squared := NormSquared(a)
	b_sqaured := NormSquared(b)
	return dp / (a_squared * b_sqaured)
}
