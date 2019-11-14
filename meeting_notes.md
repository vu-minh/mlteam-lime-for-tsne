# Meeting, Ideas, Direction, ...

## Meeting 12/11/2019

#### What we've done:
+ [X] Very simple proptotype for sampling and approximate the embedding of new points for tSNE.
	* small problem: the sampled points are "too local" (the embedding of them are always centered around the selected point), the sampling process is not gauranteed to be stable.

+ [X] No competition method :D

+ [X] How LIME does evaluation:
	* explain a white box model
	* (to fix) repeatedly remove a subset of features to show how LIME explain the important features 
	* add noise features and show that LIME does not take them into account

+ [X] (to fix) How LIME does sampling (for tabular data):
	* it does global noise sampling: consider the whole dataset as a gaussian with a mean `mu` and variance `sigma`
	* generate sample for a query point `x_i` by `x_i + mu + epsilon * sigma`
	* construct a KNN and choose first `k` nearest neighbors are proposed samples for `x_i`


#### What we should do:
+ [ ] Choose the query points:
	* one point for each class
	* maybe the point in the "intersection" of different clusters zone in the visu

+ [X] Keep simple sampling strategy to obtain the embedding for the sampled points.

+ [X] Run the [linear model]() to explain the sampled points:
	* Linear model returns matrix weights `W` indicating the importance of features
	* Visualize `W`, e.g. as a 2D heatmap to see if we can find the interested zones

+ [-] Test with more tabular datasets which have clear/understandable features:
	* Country, Boston, Wine, ...

+ [0] (OPTIONAL) Think about evaluation:
	* Can try our approach (if it works) to explain a whitebox model like PCA
	* A usecase of Benoit: t-SNE returns a large, non-homogene cluster, how can we explain it. (The main focus is to explain *what does tSNE show?*)
	* Can pre-sample uniformly many points in the dataset and find explanation for each point. (The user can explore the dataset and inverstigate the embedding of tSNE).


#### What in the stock:
+ [ ] Adapte the larger of gaussian in HD according to the given perlexity 5, 10, ...
+ [ ] Adapte the larger of student-t in LD:
	* test standard_t (variance = 1)
	* increase the variance of student-t in LD to check if the new embedded points are stable
	* normalize the variance of student-t in LD to adapte the number of neighbors in LD and HD
	(e.g. perplexity in HD is 5, standard-t in LD has 30 neighbors, -> variance of student-t in LD should be `1 / sqrt(5/30)`)


## Next Meeting 13:00 15/11/2019
-> Should show the simple sampling strategy, the result of linear model to explain the samples in LD.
-> Maybe or maybe not find the interesting explanation.
