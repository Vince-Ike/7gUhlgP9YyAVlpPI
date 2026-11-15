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

The final model achieved 92.69% average accuracy using 5-fold cross-validation, exceeding the 81% target. The model also achieved 92.93% accuracy on the test dataset.

Customer Segmentation

Customer segmentation should initially focus on customers contacted during February, March, April, June, and October. The model also identified customers without a housing loan and those contacted through cellular channels as potentially useful segments to investigate. Employment and education are additional indicators that may help distinguish customers with a higher likelihood of subscribing.

Feature Importance

The top three features were month, duration, and job, accounting for approximately 70% of total feature importance. This indicates that campaign timing is the most influential factor in the model's predictions. Call duration provides an indication of customer engagement, while employment helps distinguish customer subscription patterns.

Bonus – Business Impact

The model's 71.50% recall identified 414 out of 579 actual subscribers in the test dataset. By identifying more potential subscribers, the model gives the bank an opportunity to prioritize its marketing efforts and focus on customers who are more likely to subscribe. This could improve campaign efficiency and potentially generate additional revenue.