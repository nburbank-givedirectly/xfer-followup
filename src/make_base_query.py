import sqlparse

from mappings import FLUP_TO_RES

START_DT = "2019-10-01"
END_DT = "2023-10-01"

# Columns to include from the transfers metrics table.
TRANSFER_COLS = [
    "transfer_id",
    "recipient_id",
    "country",
    # "continent",
    # "record_type_name",
    "project_name",
    # "transfer_created_date",
    # "original_scheduled_date",
    # "most_recent_attempt_date",
    "completed_date",
    # "transfer_status",
    # "discarded_duplicate",
    # "recipient_inactive_stage",
    # "transfer_inactive",
    # "transfer_completed",
    # "transfer_amount_total_full_local_estimate",
    # "transfer_amount_total_eligible_local",
    # "transfer_amount_total_discarded_local_estimate",
    # "transfer_amount_total_full_usd_estimate",
    # "transfer_amount_total_eligible_usd",
    # "transfer_amount_total_discarded_usd_estimate",
    # "transfer_amount_total_eligible_usd_current_value",
    # "transfer_amount_total_complete_local",
    # "transfer_amount_total_outstanding_local",
    # "transfer_amount_total_complete_usd",
    # "transfer_amount_total_outstanding_usd",
    # "transfer_amount_commitment_full_local_estimate",
    # "transfer_amount_commitment_eligible_local",
    # "transfer_amount_commitment_discarded_local_estimate",
    # "transfer_amount_commitment_full_usd_estimate",
    # "transfer_amount_commitment_eligible_usd",
    # "transfer_amount_commitment_discarded_usd_estimate",
    # "transfer_amount_commitment_eligible_usd_current_value",
    # "transfer_amount_commitment_complete_local",
    # "transfer_amount_commitment_outstanding_local",
    # "transfer_amount_commitment_complete_usd",
    # "transfer_amount_commitment_outstanding_usd",
    # "index_one_on_time",
    # "index_later_delay_flup",
    "recipient_age_at_contact",
    "recipient_gender",
]

# Columns to include from the field_salesforce_followup table.
FU_ONLY = [
    "How_Much_Was_Stolen__c",
    "How_Much_Spent_on__c",
    "How_Much_Was_Bribed__c",
    "How_Much_Money_Received__c",
    "How_Much_Money_Withdrawn__c",
    "Date_of_Follow_up__c",
    "Follow_up_Type__c",
]

# Columns to include from the field_salesforce_research table.
RES_ONLY = [
    "Spending_Armed_Groups__c",
    "Spending_Benefit_Girl_Child__c",
    "Spending_Categories_Business_Active_c__c",
    "Spending_Categories_Business_Other_c__c",
    "Spending_Categories_Business_c__c",
    "Spending_Categories_Other__c",
    "Spending_Farm_Produce__c",
    "Spending_Hiring__c",
    "Spending_Improving_Home_Including_Repa__c",
    "Spending_Items_Regular_Agriculture__c",
    "Spending_Livestock_Including_Vet__c",
    "Spending_Machinery_Equipment__c",
    "Spending_Marketing_Branding__c",
    "Spending_Motorcycle_Bicycle__c",
    "Spending_Moving__c",
    "Spending_Product_Certificates__c",
    "Spending_Product_Distribution__c",
    "Spending_Qat__c",
    "Spending_Rehabilitation_Fields_Propert__c",
    "Spending_Repair_Assets__c",
    "Spending_School_Transportation__c",
    "Spending_Staff_Training__c",
    "Spending_Terrorism__c",
    "Spending_Total__c",
]


def mk_base_query(filename: str) -> None:
    query = f"""
    WITH research AS
      (SELECT *
       FROM prod_silver.field_salesforce.research
       WHERE IsDeleted = FALSE
         AND Research_Checkin_Stage__c = 'FLUP'),
         follow_up AS
      (SELECT *,
              ROW_NUMBER() over(PARTITION BY Recipient__c
                                ORDER BY Date_of_Follow_up__c DESC) AS rcpnt_fu_num,
                       ROW_NUMBER() over(PARTITION BY Transfer__c
                                ORDER BY Date_of_Follow_up__c DESC) AS tfr_fu_num
       FROM prod_silver.field_salesforce.followup fu
       WHERE fu.IsDeleted = FALSE
         AND fu.Is_Successful__c = TRUE )

      SELECT fu.Id AS fu_id,
              res.Id AS res_id,
              fu.rcpnt_fu_num,
              fu.tfr_fu_num,
              ROW_NUMBER() over(PARTITION BY res.Recipient__c, fu.Id
                                ORDER BY Date_of_Follow_up__c DESC) AS res_fu_num,
            {','.join([f't.{c}' for c in TRANSFER_COLS])},
            {','.join([f'fu.{c}' for c in FU_ONLY])},
            {','.join([f'COALESCE(res.{res_col}, fu.{fu_col}) as {res_col}' for fu_col,res_col in FLUP_TO_RES.items() ])},
            {','.join([f'res.{c}' for c in RES_ONLY])}
       
       FROM prod_gold.field_metrics.transfers t
       JOIN follow_up fu ON fu.Transfer__c = t.transfer_id
       LEFT JOIN research res ON fu.Recipient__c = res.Recipient__c
       AND abs(UNIX_TIMESTAMP(fu.CreatedDate) - UNIX_TIMESTAMP(res.CreatedDate)) < 60
       WHERE t.transfer_status = 'Complete'
         AND t.recipient_inactive_stage = 0
        AND t.transfer_created_date >= '{START_DT}'
        AND t.transfer_created_date < '{END_DT}'
    """
    query = sqlparse.format(query.strip(), reindent=True, keyword_case="upper")

    with open(f"{filename}.sql", "w") as file:
        file.write(query)


def mk_val_query(filename: str) -> None:
    query = f"""
    SELECT count(*) AS num_trans,
             count(distinct transfer_id) AS transfer_ids,
             sum(case WHEN fu.CreatedDate is NULL THEN 0 ELSE 1 end) AS non_null_cnt, 
             sum(case WHEN fu.CreatedDate is NULL THEN 0 ELSE 1 end) / count(*) AS prop_overall,
             sum(case WHEN fu.What_Did_The_Recipient_Spend_On__c is NULL THEN 0 ELSE 1 end) AS spend_cat_non_null_cnt,
             sum(case WHEN fu.What_Did_The_Recipient_Spend_On__c is NULL THEN 0 ELSE 1 end) / count(*) AS spend_cat_prop_overall

    FROM prod_gold.field_metrics.transfers t
    LEFT JOIN silver.field_salesforce_followup fu
        ON fu.Transfer__c = t.transfer_id
            AND fu.IsDeleted = FALSE
            AND fu.Is_Successful__c = TRUE
    WHERE  t.transfer_completed = 1
            AND t.recipient_inactive_stage = 0
            AND t.transfer_created_date >= '{START_DT}'
           AND t.transfer_created_date < '{END_DT}'

    """

    query = sqlparse.format(query.strip(), reindent=True, keyword_case="upper")

    with open(f"{filename}.sql", "w") as file:
        file.write(query)


if __name__ == "__main__":
    mk_base_query(filename="queries/base_query")
    mk_val_query(filename="queries/prop_xfers_w_flup")
