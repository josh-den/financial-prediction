# financial-prediction
A Case Study in the Machine Learning Process Applied to Stock Trading, Financial Prediction and a Review of the Survivorship Bias and Challenges Associated with Price Prediction

# Machine Learning Engineer Nanodegree

## Capstone Project

#### Joshua Denholtz

#### June 23nd, 2018

## A Case Study in the Machine Learning Process Applied to Stock Trading, Financial Prediction and a Review of the Survivorship Bias and Challenges Associated with Price Prediction

#### I. Definition

#### Project Overview

The predictive capabilities of Machine Learning are often applied to stock trading and financial predictions (1.,2.) to benefit individual retail traders, or large firms charged with managing their client’s money.  In both cases there is a strong incentive to risk capital for economic rewards in financial markets.  However, the typical machine learning process of testing various algorithms and features, applying them to a validation set, parameterizing, and testing, are often challenging in the time series sequential data set of a financial market.

Often quantitative funds, or amateur retail traders and Data Scientists will apply their successfully back tested models to the real world, either risking real money or in a walk forward test, and be met with large losing positions.  This paper seeks to perform a case study on the Machine Learning process applied to time series data and execute a walk forward test.  IT will also seek to validate the results using dropout and a Monte Carlo Simulation(3.) where appropriate.  

The final model will be a building block for a model with a strong case to perform comparably to the SPY in terms of risk and reward.

1. Forecasting stock market movement direction with support vector machine
Wei Huang Yoshiteru Nakamori Shou-Yang Wang

2.  Machine Learning Strategies for Time Series Forecasting
Gianluca Bontempi Souhaib Ben Taieb Yann-Aël Le Borgne

3.  An Introduction to Sequential Monte Carlo Methods
Arnaud Doucet Nando de Freitas Neil Gordon

#### Problem Statement

A data set consisting of stock market price and volume data for one-minute intervals and associated augmented data features will be used to predict as a classification if the subsequent five-minute period is in the top or bottom 20% quantile.  Therefore, the labeled dataset indicates the greatest moves upwards, and greatest moves downwards in a five-minute period.  A machine learning approach will be taken to find an algorithm that predicts if the price will move higher in the next five minutes into the top 20%.  It will do this in an attempt to outperform the risk to reward ratio of the SPY (exchange traded fund tracking the S&P500).  

1)	 The risk to reward ratio must be greater than that of the SPY the final test period.  

2)	 The algorithm must be profitable for the final test period.

The approach to identifying the optimal model well be by separating the data into many validations sets, applying algorithms, parameters and features one at a time in sequence as the data is in a time series.

The final test set will be used to as a walk forward study.  This problem is a challenging classification problem, as there are no stocks included, just the chaotic price action of the SPY.

#### Metrics

There will be multiple metrics used to quantify the best choice of algorithms and feature space.  primarily accuracy for predicting the label “1,” which indicates the algorithm predicts an upward move in the top 20% quantile for the SPY, the predictions for shorting, “-1”, will be ignored.

Selecting accuracy is done for several reasons, bearing in mind our primary goal is to optimize our risk to reward ratio, we do not care about the accuracy guessing a “0,” indicating the opposite of “1,” that the algorithm does not predict an upward move to the top 20% quantile, as if it is wrong, it only means there was a missed opportunity, and no trade takes place.  Secondly accuracy is a superior measure because accuracy, or precision, will steer our process towards very rarely placing a trade, and only doing it when the model is confident, this will reduce the number of trades but make them more accurate.  As each trade has slippage and brokerage fees assumed to be $0.03 per open and close pair, the fewer trades the better.

Additionally, another metric will be profit and loss, compared to the SPY and the risk to reward ratio compared to the SPY.  Risk is defined as the maximum draw down on the account, this number is the difference between the greatest range of sequential high to low in a designated time period.  The reward is defined as the Profit/Loss.

In other words, we will evaluate the SPY and model by looking at the profit divided by the maximum drawdown.

Risk = Max Drawdown = Maximum Price – Subsequent Lowest Price

To clarify, to “beat” the market, a model doesn’t need greater profit per share, it needs greater profit for the same amount of risk per share.  By doing so, a model could use leverage while maintaining the same amount of risk as the SPY.

Theoretically the model may use leverage as the trade is simulated for every prediction of “1” and holds for five minutes.  This means that several predictions of “1” in a row what place additional orders.  This will not impact the positivity or negativity of the model, or it’s risk to reward ratio.

#### II. Analysis

#### Data Exploration

For sequential data analyzing the bulk of features will not provide sufficient analysis as the data changes over time and without bounds.

Initially there is candle pricing information Open, High, Low, Close and Period Volume Information.  Data Augmentation will be used to increase the feature space to 763 features.  These will be reviewed in more detail in the Preprocessing section.

Rather than examine the features, the label is very interesting to explore.  This is particularly true as entering a trade based on the prediction of it being in the top 20% quantile will not have a constant risk or reward ratio.

Future 5 Minute Change

All Rows	Top 20% Quantile	Not Top 20% Quantile

Mean	0.001	0.198	-0.047

Median	0.000	0.145	-0.020

Max	7.07	7.07	0.31

Minimum	-13.67	0.04	-13.67

Standard Dev	0.184	0.199	0.143

Count	1004720	195470	809250

On average there is a slight rise in price, but not enough to justify the expenses of making the trade. The median price is similar. The maximum rise was 7/share in a five minute block and a loss of over 13/share. This puts a theoretical floor on the maximum amount of losses for an incorrect guess at 13/share, although it is historically rare. There is a standard deviation of 0.18/share.

This is interesting, as prices have clearly risen over the last decade of the SPY, yet the greatest loss is over twice the greatest rise in a 5 minute period. This clearly shows that prices move down faster, upward more often.

This is very interesting. For a correct guess, we can be rewarded with a profit of .04 to 7.07 per share, meaning not all predictions are created equally. But the average correct guess will get us just shy of .20 and median .145 per share. On the other hand, a wrong guess will see us lose about .05 on average, typically .02. Interestingly, a wrong guess could be profitable, one even rewarding the wrong guess with .31! But the risk is high as the greatest loss was 13.67. Clearly the SPY moving to the top 20% quantile occurs close to 20% of the time. Using these raw numbers we can use simple Probability Theory to compute the Expected Value of a random trade. Note that this suffers survivor bias as well as past performance does not indicate future performance. 

Expected Value = ProbA x ValueA + ProbB x ValueB where A is the 20% quantile and B is not E = (20 x .197) + (80 x -.046) = .08 Note with the example fee and slippage values of opening and then closing a trade of .03, the average guess is expected to be profitable.

Exploratory Visualization
                    
Here we see the price and volume over the entire course of the data’s history.  Interestingly prices have risen considerably, but there are no obvious patterns that may be related to price change.
   
Visually we can see things that appear to be correlations, it appears as though there is a relationship between high volumes, price changes, and more frequent price moves in the five minute period.
 
Five minute price changes are often moving both upward and downward in close proximity, indicating mean reversion and momentum is a good indicator of price move.  It also shows more frequent price moves upwards, but faster price moves downward.

#### Algorithms and Techniques

#### Algorithms

•	Linear Classifier

•	Decision Tree

•	Support Vector Classifier

•	K Nearest Neighbor

•	Bagging Ensemble

•	Random Forest

•	Gaussian Naïve Bayes

These algorithms are selected because they can easily be performed with the Sci Kit Library and without the need for lots of computing power from a GPU.  Initially the default parameters will be used from Sci Kit Learn.

#### Techniques

Many techniques are used in sequence.  As with Time Series data, the earliest data must be used first and the most recent data used last.  It must be split into time-linear parts in this way.  The earliest splits are used with the first techniques below, and as the model is furtherer and more deeply developed, more recent data is used.  

At the end of the dataset, a walk forward study is completed.  The techniques below describe this approach for one sequence at a time, going forward towards the most recent data as each more techniques are used and the algorithm is furtherer developed.

Test for Computational Resources

For each algorithm, train the data on 1% of the data set. If the training takes more than 6s, the model would take over ten minutes to run, and is therefore considered too resource intensive.

From the passing algorithms we select a base model and continue testing the remaining models.

#### Basic Model

This tests each basic model using default parameters.

Parameterized Model

This tests each model with a grid search style approach of testing numerous combinations of hyper parameters.

Single Features (multiple times)

Run the model on a single respective feature, one at a time and see how it performs.  This is a method of feature selection by finding features that perform well on their own.  This is repeated over different time periods to find features that perform well across numerous sequences.

Bottom Up

This takes the selected features and attempts to add one new feature at a time, selecting the highest scoring feature combinations.  As an example, if Feature “A” is a selected feature, the next model well try a pairing of Feature A&B, A&C, A&D, A&E….  

If “C” performs best, the next attempt begins with “A” and “C” and we try to add features B,D,E… as we look for one that performs better than A&C on their own.

We repeat this process.

Feature Sets

In this section we model a combination of feature spaces, all features, features that pass various single feature checks and bottom up feature spaces.

Final Test Model

Take the final model with its feature space, and apply it to the remaining data as a walk forward test to reveal how the algorithm performs in the wild.

Benchmark

The benchmark model will be Gaussian Naïve Bayes Classifier.  The reason for this is that of the algorithms that did not require too much computing power it performed well both in terms of computation time and accuracy.  For details on resource use see Implementation.  On the first validation set, Gaussian Naïve Bayes had the second highest accuracy making it the best candidate for a benchmark model that was not going on to be parameterized for the final model.

#### III. Methodology

#### Data Preprocessing

This dataset is covering a decade of SPY price and volume data on the minute by minute time frame.  From the OHLC(Open, High, Low, Close) and volume data, 763 additional features have been created, augmenting the data set.  These features were generated by myself with recommendations from a colleague but are not holistic of all best features, as many ones are not present that may be helpful in future work such as VWAP and RSI.  The data was obtained from IQFeed.

The created features follow trading “indicators” such as price momentum, various different measures of volatility, and location within Bollinger Bands and applied to numerous time periods both larger and smaller than the labeled data set. 

The labeled dataset classifies the subsequent change in price over a five minute period into three categories.  Those below the rolling 1850-minute 20% quantile, those above the rolling 1850-minute 80% quantile, and those that are neither.  In doing so the data is partitioned into categories representing the recent larger swings upwards or downwards.  For this case study, we will only be predicting moves upward.

This is a copy of the code that intakes OLHC and period volume, and converts it into the dataset used in this case study.  None of the original data is used in the training feature space, only the augmented data.
 
Essentially it builds properties of the current candle, such as the body, the wick, their ratios, where the low and high are relative to the open and then computes historical lagging information for several time periods.  For numerous time periods that includes; the standard deviation, the price change, the minimum and maximum values, and the Bollinger band percent, then it computes the position relative to moving averages.  This can be summarized as follows.

•	HO, LO, CO, HL – candle properties, first related to high and low minus their opening price, then the candle body and wick sizes

•	O%, Bodywick – where on the wick the open occurs, and the body to wick ratio

•	Time of day

•	For OHLC, Volume, CO, HO, LO, HL, for various time frames compute the standard deviation, change in value, minimum and maximum value, range, average, 2 standard deviations in both directions, and where the current value falls between two standard deviations(b%)

•	For various time frames compute if the period volume has surged to three times its average

•	Compute for various time frames where the price is compared to a benchmark such as a 200 day moving average

As nearly all features have normalization build in to them by definition, there is no need to perform scaling.  Take for example “HO” this is the distance between Open and High prices and “LO”, the same with the Low replacing the High price, these values are somewhat normalized by taking their values difference with the Open price.  

The features will also maintain their properties relative to one another without being scaled.  If LO is very large on average and HO is not, I would not want to scale this data as it undermines the relevancy of the normally elongated LO values.  By giving normalization to the features and not scaling them the best of both worlds, normalization and feature relationships are both maintained.

#### Implementation

The first machine learning is used to identify the computational resources in time of each model.  Due to high resources, any training that takes over ten minutes is considered too resource intensive and canceled.  We begin with linear classification, support vector classification, decision tree classification, k nearest neighbor classification, gaussian naïve bayes classification, bagging classification and random forest classification.

•	Linear Classifier

o	1 loop, best of 3: 329 ms per loop

•	Decision Tree

o	1 loop, best of 3: 17.6 s per loop

•	Support Vector Classifier

o	1 loop, best of 3: 5min 36s per loop

•	K Nearest Neighbor

o	1 loop, best of 3: 450 ms per loop

•	Bagging Ensemble

o	1 loop, best of 3: 6.15 s per loop

•	Random Forest

o	1 loop, best of 3: 2.91 s per loop

•	Gaussian Naïve Bayes

o	1 loop, best of 3: 202 ms per loop

Clearly decision tree, support vector machine and bagging ensemble are too slow.  From here, the models are analyzed for their accuracy.

The following Training and Testing percentages will be used when evaluating the models and features.  Further specifics for each are below, but also referenced in detail in “Algorithms and Techniques” as seen above.

The dataset must ensure its sequential properties, so there can be no cross validation.  A small subset of data from the first one to two years is partitioned as the first training and validation set.  From here numerous algorithms are reviewed using the full feature space.    Our scoring function will be Profit/Loss over the period.

Given the best performing algorithm, a grid search-like function will be built to test parameters without the use of cross validation to secure the sequential properties of the data.  Optimal parameters will be found by applying the models to a new validation of approximately another year where all previous data is used for training.

Feature Engineering will be performed on the subsequent ~3 years of data.  For each year, every individual feature will be used as the feature space to evaluate to Profit/Loss.  For profitable features, they will be validated again on the subsequent year, until we are left with only a few consistently profitable features.
 
For remaining features, each original feature will be paired with a different feature to improve validation performance.  If such a feature pairing is found a third feature will be attempted in the same way and so on until an ideal feature space is reached.  This is the Build Up technique described above.

At this point there are still several years of unseen data.  The current best model is reviewed against the SPY for individual years.  Following this, unseen data is used with the model to justify if it is a profitable and consistent predictor or not.

A profitable algorithm would see consistent results. But we also desire to evaluate this methodology from the survivorship bias of machine learning when applied to time series data.  Inconsistent and unprofitable results are not enough information to prove there is survivorship bias in the final algorithm.  For that reason, a Monte Carlo simulation or dropout methodology will be used where appropriate to evaluate the walk forward test and if it is consistent.

A specific function was created to take in an algorithm and feature space, and to train and predict the results, then plot them along the SPY performance allowing an easy view of the performance, risk and reward.


#### Addendums to the Original Proposal

There were several unintended changes to the original proposal.

1.	Originally evaluation criteria was F_beta score with a beta of .5.  This has been replaced with accuracy score.

a.	This was done when it was clear, even modest recall values were negatively impacting profitability and risk to reward-frequent trading triggers high brokerage fees that eat into profits.

2.	Originally Linear Classifier was selected as the base model.  This has been replaced with Gaussian Naïve Bayes.

a.	This was done because the linear classifier was predicting, for every attempt that the SPY would NOT move into the 20% quantile.  Essentially it was always guessing “0” because that occurred 80% of the time.  This would not be suitable for a base model because it would never enter the market.  Gaussian Naïve Bayes was selected because it was the next quickest model to run.

3.	Dropout is included as an alternative to Monte Carlo Simulations as the weaknesses of generated data became apparent.

a.	Monte Carlo Simulation is very good if you are looking at individual features and have lots of computing power, but not if you have many features in which your computational resources cannot simulate the non random patterns that occurs within said features.  

i.	Monte Carlo Simulation simulates data by setting volatility and return the same and generating data with similar values, but it will not capture the relationship between price and volume, and candle property relationships without simulating an entire orderbook which is very resources intensive.

ii.	The Monte Carlo Simulation would have worked however if our feature space was formed with values such as “Close_5min_momentum”, “Close_2min_monentum”,”Close_195min_std”

#### Refinement

#### Initially Algorithms

•	Linear Classifier

•	Decision Tree

•	Support Vector Classifier

•	K Nearest Neighbor

•	Bagging Ensemble

•	Random Forest

•	Gaussian Naïve Bayes

Passing Test for Computational Resources and Base Model Results

•	Linear Classifier

o	Only guesses “0”, so cannot be used going forward

•	K Nearest Neighbors

•	Random Forest

o	Highest Accuracy, selected as underlying model to carry on in developing

•	Gaussian Naïve Bayes

o	Fastest Model

o	For fastest model, and strong performing accuracy comparatively, this is selected as the base model

Param Models

Random Forest was tested in a grid search like function for n_estimators of 3,5,7,10 and max_depth of 3,5,7,10, None.  These combinations are made and used on two separate validation sets, the results are averaged.  The highest average scoring accuracy is selected, a Random Forest with a max depth of 3 with 5 estimators.

Feature Space – All Features(not including the initial features prior to data augmentation)

Over the course of three periods, a process was taking to evaluate features for selection that maintain an accuracy consistently above the score of the entire feature space.  In period 1, all features are checked respectively.  In period 2 only features passing period 1 were checked, and in period 3 only features passing period 2 were selected individually for review.  The features below consistently beat the accuracy of the model using all features over a ~3 year period.

•	Period Volume_min_5 

•	Open_bPercent_15 

•	Period Volume_momentum_195

Build Up

In build up, the previously selected features are taken one at a time, and each of them is paired with each one of the features passing period 1 of feature selection, the highest accuracy combination is added to the initial feature, and this process is iterated until there are feature spaces with a perfect accuracy, or are no longer rising.  These combinations of feature space are summarized below.

•	Period Volume_min_5, CO_momentum_5, CO

•	Period Volume_min_5, CO_momentum_5, Period Volume_lowerbound_2

•	Open_bPercent_15, CO_sma_2

•	Open_bPercent_15, Period Volume_momentum_130

•	Period Volume_momentum_195, Low_momentum_30, Close_bPercent_30

Feature Sets

For the next sequence of validation, various feature combinations are tested, with the highest score being selected.  These combinations are summarized below.

•	All features

•	All features selected from period 1

•	All features selected form period 2

•	All features selected from period 3

•	Individual features(3) selected from period 3

•	The features(5) from Build Up

Final Algorithm

From the above Feature Sets, the highest accuracy model was selected and it is summarized below.

Random Forest

•	Max Depth = 3

•	Number of Estimators = 5

Feature Space – All Features

Techniques Summarized

•	Grid Search

o	No k-fold cross validation for sequential data

•	Feature Selection

•	Feature Engineering

#### IV. Results

#### Model Evaluation and Validation

The walk forward test on the model is reasonable as expected and reaches the target solution.  Below is the results of the final model.

This hits the goals of both being profitable and with a lower risk to reward ratio.  In the walk forward study the risk to reward is about 1, but the final model gets as low as about .2!

In the proposal the case study intended to use a Monte Carlo Simulation.  After many trials, it is clear that there is not utility in a Monte Carlo Simulation for a model with a diverse feature space of this magnitude, and limited computing resources.

The reason is that machine learning models find patterns, but random data generated from the Monte Carlo Simulation does not include the truth patterns and relationships between the input features.  Take for example the Period Volume and 5 minute Momentum.  The average Period Volume is ~420k when the label of an upward move occurs, whereas without it is ~315k.  Generating random Period Volumes well not capture the fact that increased volume tends to move prices more often.  Or 5 minute Momentum has an average negative value when the label of an upper value occurs and a positive value at other times, this pattern will also not be captured.

Monte Carlo simulations are popular but are best used when there is one dimension of price action(“Close”, not the entire candle) and price action determines the entire feature space.  For a better simulation, an order book should be simulated and its results tracked, however a walk forward study is still the best way to know how the model performs.

The Monte Carlo Simulation I built generates Open prices based on volatility and returns, then selects random candle and volume properties based on their average and standard deviation.  A snippet of the function is attached here.

The interesting result of this is that the model will not know when to make a prediction because the training data, which is realistic, is not predicting anything when testing on the randomly generated data.  The results can be seen here.

This makes the model always break even as no guesses are made.  It does this over the course of many simulations.

As an alternative dropout is used by removing 20% of the training data randomly.  This will adjust the model and help to avoid overfitting.  Here is how the original model compares to the dropout model.
 
This shows model is having consistent predictions, as dropout is a strong means to reduce overfitting.  Both perform very similarly.  In fact, the dropout model performs better with a risk to reward ratio below .1!

An important question is how well does the model perform in the wild, will it?  This is impossible to answer, particularly with time series data, events markets and patterns can all change, but it can be said conclusively the model that was developed did continue to perform well in the walk forward study, which does mean it will have strong potential to succeed in the wild.  But with financial markets, there is not guarantee that past performance is indicative of future performance, no matter how well it performs for a time.

#### Justification

To justify the model is performing better than the benchmark model, see below, it compares the Gaussian Naïve Bayes Model and the Random Forest Model for the walk forward study and the walk forward study with dropout.
 
In the walk forward study, the base and final model and compared.  Both without and with dropout the final model is profitable, unlike the base model, but also has a risk to reward ratio better than the SPY.  Additionally, this shows that dropout also improves results.

This model hit all the goals in the Problem Statement, therefore the problem is solved.  It is hard to say though how will long it will continue to work in the wild, as patterns change over time.

#### V. Conclusion

#### Free-Form Visualization

The ideal model has a positive risk to reward ratio that is as close to zero as can be.

Walk Forward Study

Profit/Loss	Max Drawdown	Risk to Reward Ratio	Score

SPY	33.63	33.41	0.99	-

Base Model	-9.31	62.14	-6.68	0.37

Final Model	4.28	1.08	0.25	0.60

Clearly we can see the results of the walk forward study have performed the way intended.

#### Reflection

This was a fascinating study on one potential machine learning strategy applied to stock trading.  The study utilized a model taking every chance to avoid overfitting and often has a better risk to reward ratio than the SPY, making it a candidate for a leveraged investment strategy that could outperform the SPY.

It is interesting to note that the optimal model used the entire feature space, not an engineered feature space.  This is interesting as I suspect using few features creates a biased model.  It is very likely that there is noise in this model despite the full feature space performing well.

Additionally, it may actually be a good thing that the full feature space was used as it had been tested early on in the data set and was one of the first tested models, this made more of the tests act like a walk forward test, further validating the model.

The biggest challenge with this project was the CPU continuously ran out of memory and I would have to reload all of the data.

The results of this model fit my expectations, in some cases performing better than expected.  I believe in collaboration with financial experts this model could be improved upon and utilized in the real world, however any model should start usage with paper trading.

#### Improvement

A superior solution absolutely exists from the model found in this case study.  The primary reason is that many values were selected arbitrarily when building the dataset.  Here is a list of possible changes that can be improved upon.

•	The label is selected as 5 minutes into the future

o	Why set to five minutes?  An analytical study should be done to look for what time frame has the largest moves upward in the most predictable manner.  Additionally, there could be other exit strategies that perform better

•	The simulation of performance used Market Orders that tend to have large slippage than other orders.

o	Perhaps there is a different exit order that will perform better

•	The label classified the data by top 20% quantile

o	Why is this set to 20%?  An analytical study should be done to look for what quantile has the largest moves upward in the most predictable manner.

•	The classified label used a rolling 1850-minute quantile

o	Why is this set to 1850-minutes?  An analytical study should be done to look for what rolling time frame has the largest moves upward in the most predictable manner.

•	The dataset could have superior features added

o	VWAP is considered a strong indicator and was not included

o	RSI is considered a strong indicator and was not included

o	Candle Pattern Properties are strong indicators and were not included

o	Other features may be added

•	The feature space likely has noise

o	To reduce noise Feature Engineering could be done in which each feature with the exception of a single one is modeled for each single feature, the greatest absent feature will be removed from the feature space, one at a time narrowing down the feature space until a feature space is found with less noise.

•	GPU required deep learning models can be used

o	Due to computational resources neural networks were not included.  There has been lots of research as to the effectiveness of Recurrent Neural Networks in sequential data, particularly Long Short Term Memory models

I am confident that by improving this work, a superior final model can be made.
