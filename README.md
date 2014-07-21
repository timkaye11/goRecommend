## Go Recommend 

> Recommendation algorithms (Collaborative Filtering) in Go! 

![](http://progressed.io/bar/70)

### Background 
Collaborative Filtering (CF) is oftentimes used for item recommendations for users, and many libraries exist for other languages (popular implementations include Mahout, Prediction.IO, Apache MLLib ALS etc..). As there are very few machine learning packages out there for [Go](http://www.golang.org), I decided to put together some model based CF algorithms that I thought were interesting. Programming in Go has many benefits - namely speed/efficiency, but also the bare-bones nature allows for more customization and makes ML programming less black-box-y. 

### Collaborative Filters inside this package

- [Alternating Least Squares](http://labs.yahoo.com/files/HuKorenVolinsky-ICDM08.pdf) for both the Implicit and Explicit Case
	* Tests not yet complete
	* Use the implicit case for a confidence rating; explicit for predicting ratings
- Simple Bayesian CF Algorithm <- inside bayesianFilter/
- Similarity (using correlation, cosine and jaccard similarity) based CF, which incorporates a nearest neighbor type metric can be found in the CF folder.
	* Not complete yet. More soon! 

*Most* of the recommendation algorithms in this package are briefly outlined in [this article](http://www.hindawi.com/journals/aai/2009/421425/)


#### Additional

If you have any questions/comments, *please* feel free to reach me at tim [dot] kaye [at] lytics [dot] io




