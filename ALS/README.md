### Alternating Least Squares (ALS)

![](http://progressed.io/bar/100)

> Uses the ALS algorithm outlined in [this article](http://wanlab.poly.edu/recsys12/recsys/p83.pdf) for the explicit case. 
For the implicit case, the algorithm is outlined [here](labs.yahoo.com/files/HuKorenVolinsky-ICDM08.pdf)

Relies on Skelter John's *matrix.go* package for some matrix functionality. 

#### Usage

1. Install
```go get github.com/timkaye11/goRecommend/ALS```

2. Download the dependencies
	 - If [GPM](http://www.github.com/pote/gpm) is not installed, then do:
	 ``` brew install gpm```
	 - Get the dependencies:
	 ``` gpm install```

3. Run Code
```go
import "github.com/timkaye11/goRecommend/ALS"

func main() {
	// For this instance, cols indicate product ID ; rows indicate user ID. Remember indexing starts at 0. 
	Q := MakeRatingMatrix([]float64{
		5, 5, 5, 0, 1,
		0, 0, 0, 4, 1,
		1, 2, 3, 3, 1,
		2, 0, 4, 1, 0,
		5, 2, 0, 1, 0}, 5, 5)

	// OR load in through a text file
	// Q := Load("path/to/file", "separator") // where separator can be a comma, tally, tab, etc...

	// Train a model with 5 factors, 10 iterations, and a lambda value of 0.01.
	// 10 iterations is usually enough to reach convergence, and a lambda val of 0.01 is acceptable.
	n_factors := 5
	n_iterations := 10
	lambda := 0.01

	// Train Model Using Explicit ALS. This means that users rated each product on a scale
	// where 0 indicates not rated
	// Prints out the final error value.
	Qhat := Train(Q, n_factors, n_iterations, lambda)
	fmt.Println(Qhat)

	// Get Prediction for a user/product pair.
	fmt.Println(Predict(Qhat, 2, 1))

	// Get top - N recommended products based off of Qhat for a given user ID.
	// Args: Original user/product matrix, trained model, N, product names.
	// Returns []string of length N & error - if applicable
	// If Product Names is nil, then returns top indices for each user. Returns in descending order.

	products := []string{"Macy Gray", "The Black Keys", "Spoon", "A Tribe Called Quest", "Kanye West"}
	fmt.Println(GetTopNRecommendations(Q, Qhat, userID, n, products))

	// Implicit. Can do 'GetTopNRecommendations' in implicit case too. 
	R := TrainImplicit(Q, 5, 10, 0.01)
	fmt.Println(Predict(R, 1, 1))

}

```