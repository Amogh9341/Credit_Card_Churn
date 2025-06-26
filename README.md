
# Credit Card Churn Prediction(All below is just rubbish will make a new one)


1. Problem Statement
A bank need to know what set of people are probable of quitting on Credit Card. To ensure that people don't leave and avoid future lossses.

2. About Dataset
   <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>Attrition_Flag</th><th>Customer_Age</th><th>Gender</th><th>Dependent_count</th><th>Education_Level</th><th>Marital_Status</th><th>Income_Category</th><th>Card_Category</th><th>Months_on_book</th><th>Total_Relationship_Count</th><th>Months_Inactive_12_mon</th><th>Contacts_Count_12_mon</th><th>Credit_Limit</th><th>Total_Revolving_Bal</th><th>Avg_Open_To_Buy</th><th>Total_Amt_Chng_Q4_Q1</th><th>Total_Trans_Amt</th><th>Total_Trans_Ct</th><th>Total_Ct_Chng_Q4_Q1</th><th>Avg_Utilization_Ratio</th></tr></thead><tbody><tr><th>0</th><td>Existing Customer</td><td>45</td><td>M</td><td>3</td><td>High School</td><td>Married</td><td>$60K - $80K</td><td>Blue</td><td>39</td><td>5</td><td>1</td><td>3</td><td>12691.0</td><td>777</td><td>11914.0</td><td>1.335</td><td>1144</td><td>42</td><td>1.625</td><td>0.061</td></tr><tr><th>1</th><td>Existing Customer</td><td>49</td><td>F</td><td>5</td><td>Graduate</td><td>Single</td><td>Less than $40K</td><td>Blue</td><td>44</td><td>6</td><td>1</td><td>2</td><td>8256.0</td><td>864</td><td>7392.0</td><td>1.541</td><td>1291</td><td>33</td><td>3.714</td><td>0.105</td></tr><tr><th>2</th><td>Existing Customer</td><td>51</td><td>M</td><td>3</td><td>Graduate</td><td>Married</td><td>$80K - $120K</td><td>Blue</td><td>36</td><td>4</td><td>1</td><td>0</td><td>3418.0</td><td>0</td><td>3418.0</td><td>2.594</td><td>1887</td><td>20</td><td>2.333</td><td>0.000</td></tr><tr><th>3</th><td>Existing Customer</td><td>40</td><td>F</td><td>4</td><td>High School</td><td>Unknown</td><td>Less than $40K</td><td>Blue</td><td>34</td><td>3</td><td>4</td><td>1</td><td>3313.0</td><td>2517</td><td>796.0</td><td>1.405</td><td>1171</td><td>20</td><td>2.333</td><td>0.760</td></tr><tr><th>4</th><td>Existing Customer</td><td>40</td><td>M</td><td>3</td><td>Uneducated</td><td>Married</td><td>$60K - $80K</td><td>Blue</td><td>21</td><td>5</td><td>1</td><td>0</td><td>4716.0</td><td>0</td><td>4716.0</td><td>2.175</td><td>816</td><td>28</td><td>2.500</td><td>0.000</td></tr></tbody></table></div>
    a. Dataset is imbalanced number of Attrited Customers is too less than who stayed  
    
    ![image](https://github.com/user-attachments/assets/7863bb12-bb0a-4bad-9c23-f8993d725341)
   
    b. Credit Limit and Average Open to Buy are very closely co-related thus keepin gonly one of them is enough.

   ![image](https://github.com/user-attachments/assets/183757f2-38b7-41cc-b283-1c7364ae7d15)

   ```python
   # Your code here
   df.columns
   ```


