SELECT count(*) AS num_trans,
       count(DISTINCT transfer_id) AS transfer_ids,
       sum(CASE
               WHEN fu.CreatedDate IS NULL THEN 0
               ELSE 1
           END) AS non_null_cnt,
       sum(CASE
               WHEN fu.CreatedDate IS NULL THEN 0
               ELSE 1
           END) / count(*) AS prop_overall,
       sum(CASE
               WHEN fu.What_Did_The_Recipient_Spend_On__c IS NULL THEN 0
               ELSE 1
           END) AS spend_cat_non_null_cnt,
       sum(CASE
               WHEN fu.What_Did_The_Recipient_Spend_On__c IS NULL THEN 0
               ELSE 1
           END) / count(*) AS spend_cat_prop_overall
FROM common.field_metrics_transfers t
LEFT JOIN silver.field_salesforce_followup fu ON fu.Transfer__c = t.transfer_id
AND fu.IsDeleted = FALSE
AND fu.Is_Successful__c = TRUE
WHERE t.transfer_status = 'Complete'
  AND t.recipient_inactive_stage = 0
  AND t.transfer_created_date >= '2020-01-01'
  AND t.transfer_created_date < '2023-10-01'