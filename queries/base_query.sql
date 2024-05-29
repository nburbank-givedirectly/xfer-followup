WITH research AS
  (SELECT *
   FROM silver.field_salesforce_research
   WHERE IsDeleted = FALSE
     AND Research_Checkin_Stage__c = 'FLUP'),
     follow_up AS
  (SELECT *,
          ROW_NUMBER() over(PARTITION BY Recipient__c
                            ORDER BY Date_of_Follow_up__c DESC) AS rcpnt_fu_num,
          ROW_NUMBER() over(PARTITION BY Transfer__c
                            ORDER BY Date_of_Follow_up__c DESC) AS tfr_fu_num
   FROM silver.field_salesforce_followup fu
   WHERE fu.IsDeleted = FALSE
     AND fu.Is_Successful__c = TRUE )
SELECT fu.Id AS fu_id,
       res.Id AS res_id,
       fu.rcpnt_fu_num,
       fu.tfr_fu_num,
       ROW_NUMBER() over(PARTITION BY res.Recipient__c, fu.Id
                         ORDER BY Date_of_Follow_up__c DESC) AS res_fu_num,
       t.transfer_id,
       t.recipient_id,
       t.country,
       t.project_name,
       t.completed_date,
       t.recipient_age_at_contact,
       t.recipient_gender,
       fu.How_Much_Was_Stolen__c,
       fu.How_Much_Spent_on__c,
       fu.How_Much_Was_Bribed__c,
       fu.How_Much_Money_Received__c,
       fu.How_Much_Money_Withdrawn__c,
       fu.Date_of_Follow_up__c,
       fu.Follow_up_Type__c,
       fu.What_Did_The_Recipient_Spend_On_Other__c,
       COALESCE(res.Spending_Categories__c, fu.What_Did_The_Recipient_Spend_On__c) AS Spending_Categories__c,
       COALESCE(res.Spending_USAID_Banned_Items__c, fu.How_Much_Spent_on_USAID_Banned_Items__c) AS Spending_USAID_Banned_Items__c,
       COALESCE(res.Spending_Livestock__c, fu.How_Much_Spent_on_Livestock__c) AS Spending_Livestock__c,
       COALESCE(res.Spending_Agriculture__c, fu.How_Much_Spent_on_Agriculture__c) AS Spending_Agriculture__c,
       COALESCE(res.Spending_Existing_Business__c, fu.How_Much_Spent_on_Existing_Business__c) AS Spending_Existing_Business__c,
       COALESCE(res.Spending_New_Business__c, fu.How_Much_Spent_on_New_Business__c) AS Spending_New_Business__c,
       COALESCE(res.Spending_Self_Education__c, fu.How_Much_Spent_on_Self_Education__c) AS Spending_Self_Education__c,
       COALESCE(res.Spending_Other_Education__c, fu.How_Much_Spent_on_Other_Education__c) AS Spending_Other_Education__c,
       COALESCE(res.Spending_Building_Home__c, fu.How_Much_Spent_on_Building_Home__c) AS Spending_Building_Home__c,
       COALESCE(res.Spending_Sanitation__c, fu.How_Much_Spent_on_Sanitation__c) AS Spending_Sanitation__c,
       COALESCE(res.Spending_Loans_Debts__c, fu.How_Much_Spent_on_Loans_Debts__c) AS Spending_Loans_Debts__c,
       COALESCE(res.Spending_Savings__c, fu.How_Much_Spent_on_Savings__c) AS Spending_Savings__c,
       COALESCE(res.Spending_Rent__c, fu.How_Much_Spent_on_Rent__c) AS Spending_Rent__c,
       COALESCE(res.Spending_Energy__c, fu.How_Much_Spent_on_Energy__c) AS Spending_Energy__c,
       COALESCE(res.Spending_Food__c, fu.How_Much_Spent_on_Food__c) AS Spending_Food__c,
       COALESCE(res.Spending_Health__c, fu.How_Much_Spent_on_Health__c) AS Spending_Health__c,
       COALESCE(res.Spending_Household_Goods__c, fu.How_Much_Spent_on_Household_Goods__c) AS Spending_Household_Goods__c,
       COALESCE(res.Spending_Church__c, fu.How_Much_Spent_on_Church__c) AS Spending_Church__c,
       COALESCE(res.Spending_Events__c, fu.How_Much_Spent_on_Events__c) AS Spending_Events__c,
       COALESCE(res.Spending_Leisure__c, fu.How_Much_Spent_on_Leisure__c) AS Spending_Leisure__c,
       COALESCE(res.Spending_Gifts__c, fu.How_Much_Spent_on_Gifts__c) AS Spending_Gifts__c,
       COALESCE(res.Spending_Other_Items__c, fu.How_Much_Spent_on_Other_Items__c) AS Spending_Other_Items__c,
       COALESCE(res.Spending_Land__c, fu.How_much_spent_on_land__c) AS Spending_Land__c,
       COALESCE(res.Spending_Improving_Home__c, fu.How_much_spent_on_improving_home__c) AS Spending_Improving_Home__c,
       COALESCE(res.Spending_Clothes_Furniture__c, fu.How_much_spent_on_clothes_or_furniture__c) AS Spending_Clothes_Furniture__c,
       COALESCE(res.Spending_Drugs__c, fu.How_much_spent_on_drugs__c) AS Spending_Drugs__c,
       COALESCE(res.Spending_Gambling__c, fu.How_much_spend_on_gambling__c) AS Spending_Gambling__c,
       COALESCE(res.Spending_Sex__c, fu.How_much_spent_on_sex__c) AS Spending_Sex__c,
       COALESCE(res.Spending_Abortion__c, fu.How_much_spent_on_abortion__c) AS Spending_Abortion__c,
       COALESCE(res.Spending_Tools_Equipment__c, fu.How_Much_Spent_on_Tools_Equipment__c) AS Spending_Tools_Equipment__c,
       COALESCE(res.Spending_Unspent__c, fu.How_Much_Unspent__c) AS Spending_Unspent__c,
       res.Spending_Armed_Groups__c,
       res.Spending_Benefit_Girl_Child__c,
       res.Spending_Categories_Business_Active_c__c,
       res.Spending_Categories_Business_Other_c__c,
       res.Spending_Categories_Business_c__c,
       res.Spending_Categories_Other__c,
       res.Spending_Farm_Produce__c,
       res.Spending_Hiring__c,
       res.Spending_Improving_Home_Including_Repa__c,
       res.Spending_Items_Regular_Agriculture__c,
       res.Spending_Livestock_Including_Vet__c,
       res.Spending_Machinery_Equipment__c,
       res.Spending_Marketing_Branding__c,
       res.Spending_Motorcycle_Bicycle__c,
       res.Spending_Moving__c,
       res.Spending_Product_Certificates__c,
       res.Spending_Product_Distribution__c,
       res.Spending_Qat__c,
       res.Spending_Rehabilitation_Fields_Propert__c,
       res.Spending_Repair_Assets__c,
       res.Spending_School_Transportation__c,
       res.Spending_Staff_Training__c,
       res.Spending_Terrorism__c,
       res.Spending_Total__c
FROM common.field_metrics_transfers t
JOIN follow_up fu ON fu.Transfer__c = t.transfer_id
LEFT JOIN research res ON fu.Recipient__c = res.Recipient__c
AND abs(UNIX_TIMESTAMP(fu.CreatedDate) - UNIX_TIMESTAMP(res.CreatedDate)) < 60
WHERE t.transfer_status = 'Complete'
  AND t.recipient_inactive_stage = 0
  AND t.transfer_created_date >= '2019-10-01'
  AND t.transfer_created_date < '2024-01-31'