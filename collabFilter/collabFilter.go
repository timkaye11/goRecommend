package collabFilter

import (
	"errors"
	"fmt"
	. "github.com/skelterjohn/go.matrix"
	"math"
	"sort"
	"strconv"
)

func errcheck(err error) {
	if err != nil {
		fmt.Printf("\nError:  %v occured", err)
	}
}

// Find the dot product between two vectors
func DotProduct(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return float64(0), errors.New("Cannot dot vectors of different length")
	}
	prod := float64(0)
	for i := 0; i < len(a); i++ {
		prod += a[i] * b[i]
	}
	return prod, nil
}

// For cosine similarity
func NormSquared(a []float64) float64 {
	sum := float64(0)
	for i := 0; i < len(a); i++ {
		sum += a[i] * a[i]
	}
	return math.Sqrt(sum)
}

// Cosine Similarity between two vectors
func CosineSim(a, b []float64) float64 {
	dp, err := DotProduct(a, b)
	errcheck(err)
	a_squared := NormSquared(a)
	b_sqaured := NormSquared(b)
	return dp / (a_squared * b_sqaured)
}

// defined as A n B / A u B. Used for binary user/product matrices.
func Jaccard(a, b []float64) float64 {
	intersection := float64(0)
	for i := 0; i < len(a); i++ {
		if a[i] == b[i] {
			intersection += 1
		}
	}
	union := float64(0)
	for i := 0; i < len(a); i++ {
		if a[i] > 0 || b[i] > 0 {
			union += 1
		}
	}
	return intersection / union
}

func replaceNA(prefs *DenseMatrix) *DenseMatrix {
	arr := prefs.Array()
	for i := 0; i < len(arr); i++ {
		if math.IsNaN(arr[i]) {
			arr[i] = float64(0)
		}
	}
	return MakeDenseMatrix(arr, prefs.Rows(), prefs.Cols())
}

// Gets Recommendations for a user (row index) based on the prefs matrix.
// Uses cosine similarity for rating scale, and jaccard similarity if binary
func getRecommendations(prefs *DenseMatrix, user int, products []string) ([]string, []float64, error) {
	// make sure user is in the preference matrix
	if user >= prefs.Rows() {
		return nil, nil, errors.New("user index out of range")
	}
	prefs = replaceNA(prefs)
	// item ratings
	ratings := make(map[int]float64, 0)
	sims := make(map[int]float64, 0)
	// Get user row from prefs matrix
	user_ratings := prefs.GetRowVector(user).Array()
	for i := 0; i < prefs.Rows(); i++ {
		// don't compare row to itself.
		if i != user {
			// get cosine similarity for other scores.
			other := prefs.GetRowVector(i).Array()
			cos_sim := CosineSim(user_ratings, other)
			// get product recs for neighbors
			for idx, val := range other {
				if (user_ratings[idx] == 0 || math.IsNaN(user_ratings[idx])) && val != 0 {
					weighted_rating := val * cos_sim
					ratings[idx] += weighted_rating
					sims[idx] += cos_sim
				}
			}
		}
	}
	recs, vals := calculateWeightedMean(ratings, sims, products)
	return recs, vals, nil
}

func sum(x []float64) float64 {
	sum := float64(0)
	for i := 0; i < len(x); i++ {
		sum += x[i]
	}
	return sum
}

// Gets Recommendations for a user (row index) based on the prefs matrix.
// Uses cosine similarity for rating scale, and jaccard similarity if binary
func getBinaryRecommendations(prefs *DenseMatrix, user int, products []string) ([]string, []float64, error) {
	// make sure user is in the preference matrix
	if user >= prefs.Rows() {
		return nil, nil, errors.New("user index out of range")
	}
	prefs = replaceNA(prefs)
	// item ratings
	ratings := make(map[float64]string)
	// Get user row from prefs matrix
	user_ratings := prefs.GetRowVector(user).Array()

	for ii := 0; ii < prefs.Cols(); ii++ {
		if user_ratings[ii] == float64(0) {
			num_liked := sum(prefs.GetColVector(ii).Array())
			num_disliked := float64(prefs.Rows()) - num_liked
			jaccard_liked := make([]float64, 0)
			jaccard_disliked := make([]float64, 0)
			for i := 0; i < prefs.Rows(); i++ {
				if i != user {
					other := prefs.GetRowVector(i).Array()
					if other[ii] == float64(0) {
						jaccard_disliked = append(jaccard_disliked, Jaccard(user_ratings, other))
					} else {
						jaccard_liked = append(jaccard_liked, Jaccard(user_ratings, other))
					}
				}
			}
			rating := (sum(jaccard_liked) - sum(jaccard_disliked)) / (num_disliked + num_liked)
			if products != nil {
				ratings[rating] = products[ii]
			} else {
				ratings[rating] = strconv.Itoa(ii)
			}
		}
	}
	prods, scores := sortMap(ratings)
	return prods, scores, nil
}

func calculateWeightedMean(ratings, sims map[int]float64, products []string) (recommends []string, values []float64) {
	recommendations := make(map[float64]string, 0)
	for k, v := range ratings {
		mean_product_rating := v / sims[k]
		fmt.Println(mean_product_rating)
		if products != nil {
			recommendations[mean_product_rating] = products[k]
		} else {
			recommendations[mean_product_rating] = strconv.Itoa(k)
		}
	}
	recommends, values = sortMap(recommendations)
	return
}

// Sorts a map of floats -> strings to get best recommendations. Probably a better way to do this.
func sortMap(recs map[float64]string) ([]string, []float64) {
	vals := make([]float64, 0)
	for k, _ := range recs {
		vals = append(vals, k)
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(vals)))
	prods := make([]string, 0)
	for _, val := range vals {
		prods = append(prods, recs[val])
	}
	return prods, vals
}

func MakeRatingMatrix(ratings []float64, rows, cols int) *DenseMatrix {
	return MakeDenseMatrix(ratings, rows, cols)
}
