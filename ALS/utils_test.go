package ALS

import (
	"testing"
)

func LoadTest(t *testing.T) {
	// load in the test data with the separator as a comma
	relationship_matrix := Load("../testdata/data.txt", ",")

	Assert(t, relationship_matrix.Rows() == 4)
	Assert(t, relationship_matrix.Cols() == 5)
	Assert(t, relationship_matrix.Get(0, 4) == 1)
}
