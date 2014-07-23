### Bayesian Collaborative-Filtering in Go

Inspired by [this link](http://www.hindawi.com/journals/aai/2009/421425/) and uses Skelterjohn's [go.matrix](http://github.com/skelterjohn/go.matrix) package.

Bayesian CF is a model based recommendation algorithm and utilizes a laplace smoother. 

To use:
 - Install: ``` go get github.com/timkaye11/goCF/bayesianFilter```
 -  Pass in a dense matrix with user and item indices:
```
import (
	"github.com/skelterjohn/go.matrix"
	. "github.com/timkaye11/goCF/bayesianFilter"
)

func main() {

	mat := MakeRatingMatrix([]float64{
		4, NA, 5, 5,
		4, 2, 1, NA,
		3, NA, 2, 4,
		4, 4, NA, NA,
		2, 1, 3, 5}, 5, 4)
	
	// 
	_, predictedValue, err := BayesianFilter(mat, 0, 1)
	// do something with err
	fmt.Println(predictedValue)

}
```

---
 
#### More Info
Some more information about BCF can be found at
 - [A Bayesian Model for Collaboritive Filtering](http://www-stat.wharton.upenn.edu/~edgeorge/Research_papers/Bcollab.pdf)
 - [Collaborative Filtering With a Simple Bayesian Classifier](http://www.ics.uci.edu/~pazzani/Publications/IPSJ.pdf)


