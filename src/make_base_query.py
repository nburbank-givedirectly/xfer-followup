import sqlparse

from col_lsts import TRANSFER_COLS, FU_ONLY, RES_ONLY, FLUP_TO_RES

START_DT = "2020-01-01"
END_DT = "2023-10-01"


def mk_base_query(filename: str) -> None:
    query = f"""
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
            {','.join([f't.{c}' for c in TRANSFER_COLS])},
            {','.join([f'fu.{c}' for c in FU_ONLY])},
            {','.join([f'COALESCE(res.{res_col}, fu.{fu_col}) as {res_col}' for fu_col,res_col in FLUP_TO_RES.items() ])},
            {','.join([f'res.{c}' for c in RES_ONLY])}
       
       FROM common.field_metrics_transfers t
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
             sum(case WHEN fu.CreatedDate is NULL THEN 1 ELSE 0 end) AS non_null_cnt, 
             sum(case WHEN fu.CreatedDate is NULL THEN 1 ELSE 0 end) / count(*) AS prop_overall,
             sum(case WHEN fu.What_Did_The_Recipient_Spend_On__c is NULL THEN 1 ELSE 0 end) AS spend_cat_non_null_cnt,
             sum(case WHEN fu.What_Did_The_Recipient_Spend_On__c is NULL THEN 1 ELSE 0 end) / count(*) AS spend_cat_prop_overall

    FROM common.field_metrics_transfers t
    LEFT JOIN silver.field_salesforce_followup fu
        ON fu.Transfer__c = t.transfer_id
            AND fu.IsDeleted = FALSE
            AND fu.Is_Successful__c = TRUE
            AND fu.Has_Transfer_Been_Collected__c = TRUE
    WHERE t.transfer_status = 'Complete'
            AND t.recipient_inactive_stage = 0
            AND t.transfer_created_date >= '{START_DT}'
           AND t.transfer_created_date < '{END_DT}'

    """

    query = sqlparse.format(query.strip(), reindent=True, keyword_case="upper")

    with open(f"{filename}.sql", "w") as file:
        file.write(query)


if __name__ == "__main__":
    mk_base_query(filename="../queries/base_query")
    mk_val_query(filename="../queries/val_query")
