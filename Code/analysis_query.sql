use MultSeq;
with exec_stats as (
    select
        cloud_execution,
        ROUND(avg(fdp),3) as fdr,
        ROUND(avg(fnp),3) as fnr,
        ROUND(avg(avg_sample_number),3) as asn,
        count(num_not_terminated > 0 or NULL) as num_with_unterminated,
        count(*) as num_mcs,
        ROUND(SUM(fdp) / count(
            fdp > 0
            or NULL
        ),3) as pfdr,
        ROUND(SUM(fnp) / count(
            fnp > 0
            or NULL
        ),3) as pfnr,
        ROUND(STDDEV(fdp),3)  as var_fdr,
        ROUND(STDDEV(fnp),3)  as var_fnr,
        ROUND(STDDEV(avg_sample_number),3)  as var_asn,
        ROUND(avg(num_not_terminated),3)  as avg_num_not_terminated,
        ROUND(STDDEV(num_not_terminated) ,3) as var_num_not_terminated
    from
        simulation_results
    group by
        cloud_execution
),
relevant_cloud as (
    select
        cloud_execution,
        run_start
    from
        simulation_metadata
    where
        sim_reps = 1000
        and run_start > "2024-04-01"
),
cloud_params as (
    Select * from simulation_params inner join relevant_cloud using (cloud_execution)
)
select
    *
from
    exec_stats
    inner join cloud_params using (cloud_execution)
order by
    run_start;