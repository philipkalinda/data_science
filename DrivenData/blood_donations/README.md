### Warm Up: Predict Blood Donations



#### Predicting Blood Donations
We've all got to start somewhere. 
This is the smallest, least complex dataset on DrivenData. 
That makes it a great place to dive into the world of data science competitions. 
Get your blood pumping and try your hand at predicting donations.



#### Blood Donations
Blood donation has been around for a long time.
The first successful recorded transfusion was between two dogs in 1665, and the first medical use of human blood in a transfusion occurred in 1818. 
Even today, donated blood remains a critical resource during emergencies.



#### Red Cross, 1943

Our dataset is from a mobile blood donation vehicle in Taiwan. 
The Blood Transfusion Service Center drives to different universities and collects blood as part of a blood drive. 
We want to predict whether or not a donor will give blood the next time the vehicle comes to campus.



#### About Donating Blood

Here at DrivenData, we believe donating blood is important. Good data-driven systems for tracking and predicting donations and supply needs can improve the entire supply chain, making sure that more patients get the blood transfusions they need.

In the United States, the American Red Cross is a good resource for information about donating blood. According to their website:
•Every two seconds someone in the U.S. needs blood.

•More than 41,000 blood donations are needed every day.

•A total of 30 million blood components are transfused each year in the U.S.

•The blood used in an emergency is already on the shelves before the event occurs.

•Sickle cell disease affects more than 70,000 people in the U.S. About 1,000 babies are born with the disease each year. Sickle cell 
patients can require frequent blood transfusions throughout their lives.

•More than 1.6 million people were diagnosed with cancer last year. Many of them will need blood, sometimes daily, during their chemotherapy treatment.

•A single car accident victim can require as many as 100 pints of blood.


For more information, you can look at the website of the American Red Cross: http://www.redcrossblood.org/donating-blood/why-donate-blood



#### The Blood Transfusion Service Center Dataset

The UCI Machine Learning Repository is a great resource for practicing your data science skills. They provide a wide range of datasets for testing machine learning algorithms. Finding a subject matter you're interested in can be a great way to test yourself on real-world data problems. Given our mission, we're interested in predicting if a blood donor will donate within a given time window.

Here's what the first few rows of the training set look like:

Months since Last Donation

Number of Donations

Total Volume Donated (c.c.)

Months since First Donation

Made Donation in March 2007

Predict if the donor will give in March 2007

The goal is to predict the last column, whether he/she donated blood in March 2007.

Use information about each donor's history
•Months since Last Donation: this is the number of monthis since this donor's most recent donation.
•Number of Donations: this is the total number of donations that the donor has made.
•Total Volume Donated: this is the total amound of blood that the donor has donated in cubuc centimeters.
•Months since First Donation: this is the number of months since the donor's first donation.



#### Submission format

This competitions uses log loss as its evaluation metric, so the predictions you submit are the probability that a donor made a donation in March 2007.

The submission format is a csv with the following columns: 

Made Donation in March 2007

To be explicit, you need to submit a file like the following with predictions for every ID in the Test Set we provide:
,Made Donation in March 2007
659,0.5
276,0.5
263,0.5
303,0.5
...



#### Data citation

Data is courtesy of Yeh, I-Cheng via the UCI Machine Learning repository:

Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, "Knowledge discovery on RFM model using Bernoulli sequence, "Expert Systems with Applications, 2008, doi:10.1016/j.eswa.2008.07.018.
