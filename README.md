# 7gUhlgP9YyAVlpPI
TD Marketing
Goal:

Predict if the customer will subscribe (yes/no) to a term deposit 

Success Metric(s):

Hit 81% or above accuracy by evaluating with 5-fold cross validation and reporting the average performance score.

Current Challenges:

We are also interested in finding customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.

What makes the customers buy? Tell us which feature we should be focusing more on.

Bonus from mentor:

think about how i can correlate the results to a business metric
for example revenue 

Results Summary

Success Metric Achieved

The final XGBoost model achieved 92.69% average accuracy using 5-fold cross-validation, exceeding the project target of 81%. After optimizing the classification probability threshold to 0.53, the model achieved 93.23% accuracy on the test dataset, with 52.46% precision, 68.22% recall, and a 59.31% F1 score.

Customer Segmentation

Customer segmentation should initially focus on customers contacted during February, March, April, June, and October, as these months had the greatest influence on the model's predictions. The model also identified customers without a housing loan and those contacted through cellular channels as potentially useful segments to investigate. Employment and education are additional indicators that may help distinguish customers with a higher likelihood of subscribing.

These findings can help guide future marketing campaigns, although the identified patterns should be validated against actual subscription rates before making targeting decisions.

Feature Importance

The top three features were month, duration, and job, accounting for approximately 70% of total feature importance. This indicates that campaign timing is the most influential factor in the model's predictions. Call duration provides an indication of customer engagement, while employment helps distinguish customer subscription patterns.

However, duration is only available after a customer has been contacted. Therefore, it is more useful for evaluating customer engagement during or after a campaign than for identifying prospects before the initial call.

Bonus – Business Impact

Using the optimized probability threshold of 0.53, the model achieved 68.22% recall, correctly identifying 395 of the 579 actual subscribers in the test dataset. It also identified 358 customers who were predicted to subscribe but did not, resulting in 753 customers being selected as potential prospects.

By identifying customers with a higher likelihood of subscribing, the model gives the bank an opportunity to prioritize its marketing efforts and allocate resources more efficiently. The selected group had a 52.46% subscription rate, compared with an overall test-dataset subscription rate of approximately 7.24%. This represents roughly 7.25 times the overall subscription rate and suggests that the model could help improve campaign targeting and potentially increase revenue.

Actual financial impact would need to be validated through campaign results, conversion tracking, and the cost and value of acquiring each subscriber.