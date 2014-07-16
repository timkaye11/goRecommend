package main // ALS

import (
	"errors"
	"fmt"
	. "github.com/skelterjohn/go.matrix"
	"math"
	"math/rand"
)

var (
	NA     = math.NaN()
	lambda = 0.1
)

func errcheck(err error) {
	if err != nil {
		fmt.Printf("Error occured: %v", err)
	}
}

// Create the W matrix for the ALS algorithm..
func MakeWeightMatrix(mat *DenseMatrix) *DenseMatrix {
	values := mat.Array()
	newvalues := make([]float64, len(values))
	for i := 0; i < len(values); i++ {
		if values[i] == 0.0 || math.IsNaN(values[i]) {
			newvalues[i] = 0
		} else {
			newvalues[i] = 1
		}
	}
	return MakeDenseMatrix(newvalues, mat.Rows(), mat.Cols())
}

// create X and Y matrices for the ALS algorithm
func MakeXY(mat *DenseMatrix, n_factors int) (X, Y *DenseMatrix) {
	rows := mat.Rows()
	cols := mat.Cols()
	X_data := make([]float64, rows*n_factors)
	Y_data := make([]float64, cols*n_factors)
	for i := 0; i < len(X_data); i++ {
		X_data[i] = 5 * rand.Float64()
	}
	for j := 0; j < len(Y_data); j++ {
		Y_data[j] = 5 * rand.Float64()
	}
	X = MakeDenseMatrix(X_data, rows, n_factors)
	Y = MakeDenseMatrix(Y_data, n_factors, cols)
	return
}

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

// adds up all the elements of the array
func SumMatrix(mat *DenseMatrix) (sum float64) {
	values := mat.Array()
	sum = float64(0)
	for i := 0; i < len(values); i++ {
		sum += values[i]
	}
	return
}

// Scales matrix mat by weight.
func SimpleTimes(mat, weight *DenseMatrix) *DenseMatrix {
	if len(mat.Array()) != len(weight.Array()) {
		return nil
	}
	weightArray := weight.Array()
	matValues := mat.Array()
	for i := 0; i < len(matValues); i++ {
		matValues[i] = weightArray[i] * matValues[i]
	}
	return MakeDenseMatrix(matValues, mat.Rows(), mat.Cols())
}

// Gets the error for the alternating least squares algorithm
func GetError(W, Q, X, Y *DenseMatrix) float64 {
	dot, err := X.TimesDense(Y)
	errcheck(err)
	err = Q.SubtractDense(dot)
	errcheck(err)
	Prod := SimpleTimes(Q, W)
	tosum := SimpleTimes(Prod, Prod)
	sum := SumMatrix(tosum)
	return sum
}

// Alternating Least Sqaures for Collaborative Filtering
func ALS(Q, W, X, Y *DenseMatrix, iterations, n_factors int) *DenseMatrix {
	for i := 0; i < iterations; i++ {
		I := Eye(n_factors)
		I.Scale(lambda)

		Y_dot, err := Y.TimesDense(Y.Transpose())
		errcheck(err)
		Y_dot.AddDense(I)
		X_toSolve, _ := Y.TimesDense(Q.Transpose())
		//X, _ = Y_dot.SolveDense(X_toSolve)
		X = MatrixSolver(Y_dot, X_toSolve)
		X = X.Transpose()

		X_dot, err := X.Transpose().TimesDense(X)
		errcheck(err)
		X_dot.AddDense(I)
		Y_toSolve, _ := Q.TimesDense(X.Transpose())
		//Y, _ = X_dot.SolveDense(Y_toSolve)
		Y = MatrixSolver(X_dot, Y_toSolve)
	}
	Q_hat, _ := X.TimesDense(Y)
	return Q_hat

}

// Auxilliary function for Matrix Solver.
func SwapCols(mat *DenseMatrix, i, j int) *DenseMatrix {
	p := mat.Copy()
	trans := p.Transpose()
	trans.SwapRows(i, j)
	toret := trans.Transpose()
	return toret
}

// solves AX = B using matrix inversion. Utilizes the Solve method from
// Skelter John's matrix package, and creates the X matrix s.t AX = B.
func MatrixSolver(mat, outcome *DenseMatrix) *DenseMatrix {
	rows := make([]float64, 0)
	firstRow, _ := mat.SolveDense(outcome)
	rows = append(rows, firstRow.Array()...)
	for i := 1; i < mat.Cols(); i++ {
		matrix := mat.Copy()
		swapped := SwapCols(outcome, 0, i)
		value, _ := matrix.SolveDense(swapped)
		rows = append(rows, value.Array()...)
	}
	return MakeDenseMatrix(rows, mat.Rows(), mat.Cols())
}

func main() {
	// 0's indicate NA's
	// For this instance, cols indicate movie ID ; rows indicate userID
	Q := MakeDenseMatrix([]float64{5, 5, 5, 5, 1,
		0, 0, 0, 4, 1,
		1, 2, 3, 3, 1,
		2, 2, 2, 1, 0,
		5, 2, 5, 1, 0}, 5, 5)

	W := MakeWeightMatrix(Q)
	//X, Y := MakeXY(Q, 5)
	XT := MakeDenseMatrix([]float64{1.85573851, 0.7969648, 0.56948032, 2.7325095, 3.26833747,
		2.0448829, 4.98801804, 4.99830168, 2.69795169, 3.5537049,
		4.14321262, 1.38068421, 4.18672857, 1.56950527, 1.72737021,
		4.13532677, 3.89585519, 0.73849721, 1.25576478, 0.07823237,
		1.39345008, 2.57900365, 2.63576582, 3.91586586, 3.35085364}, 5, 5)
	YT := MakeDenseMatrix([]float64{0.14719139, 4.00940444, 3.3813894, 3.87346365, 0.90669916,
		0.83974263, 1.73973142, 0.16262076, 3.91060583, 1.81874077,
		3.92818648, 3.38339231, 0.99215264, 3.10931533, 1.44353955,
		4.11526834, 0.70579904, 3.23214613, 3.77848242, 3.21096918,
		2.25101798, 2.5705109, 2.66376777, 2.17691912, 0.25290882}, 5, 5)

	fmt.Println(GetError(W, Q.Copy(), XT, YT))
	//	val, _ := XT.TimesDense(YT)

	fmt.Println(ALS(Q, W, XT, YT, 1, 5))

	I := Eye(5)
	I.Scale(lambda)

	Y_dot, err := YT.TimesDense(YT.Transpose())
	errcheck(err)
	Y_dot.AddDense(I)
	X_toSolve, _ := YT.TimesDense(Q.Transpose())
	XT, _ = Y_dot.SolveDense(X_toSolve)
	//fmt.Println(Y_dot)
	fmt.Println(X_toSolve)
	XT = XT.Transpose()
	//fmt.Println(XT)
	fmt.Println("\n\n")
	fmt.Println(MatrixSolver(Y_dot, X_toSolve))

}
