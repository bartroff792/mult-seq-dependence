import os
import shelve
import tqdm
from contextlib import closing
import pandas
import configparser
from utils.sim_analysis import est_equiv_sample_size
from numpy import log

def get_ave_samp_ser(fullrec, hyp_type, m0):
    return pandas.Series({
                 "H0 ASN":fullrec[hyp_type].iloc[:,:m0].mean().mean(),
                 "Ha ASN":fullrec[hyp_type].iloc[:,m0:].mean().mean(),
                 "Seq Achieved FDR":fullrec["FDR"][hyp_type]["rej_FDR"],
                 "Seq Achieved FNR":fullrec["FDR"][hyp_type]["rej_FNR"],
                 "Nominal FDR":fullrec["config"].getfloat("alpha_general"),
                 "Nominal FNR":fullrec["config"].getfloat("beta_general")})
                                                                    

def term_time_general(shfp, cfgfp, cfgsect):
    """Retrieve termination time and error metric of seq table from shelve file
    """
    rec_filepath = os.path.expanduser(shfp)
    with closing(shelve.open(rec_filepath)) as shf:
        FDR_table = shf["FDR_table"]
        seq_pois_data_general = shf["synth_pois_data_general"]
        seq_binom_data_general = shf["synth_binom_data_general"]
        
    config = configparser.ConfigParser(inline_comment_prefixes=["#"], default_section="default")
    config.read([os.path.expanduser(cfgfp)])
    config_section = config[cfgsect]
    m0 = config_section.getint("m_null")
    m1 = config_section.getint("m_alt")
    print(config_section.getint("n_periods_rejective"))
    fullrec = {"pois":seq_pois_data_general[1], "binom":seq_binom_data_general[1], "FDR":FDR_table, "config":config_section}
    fullrec["ave_samp_pois"] = get_ave_samp_ser(fullrec, "pois", m0)
    fullrec["ave_samp_binom"] = get_ave_samp_ser(fullrec, "binom", m0)
    return ((m0, m1), fullrec)

cfg_sects = ["bl55", "bl37", "bl73"]
shelve_pattern = "~/Dropbox/Research/MultSeq/Data/{0}.shelve"
config_path = "~/Dropbox/Research/MultSeq/Data/sim.cfg"
raw_list = [ term_time_general(shelve_pattern.format(cfg_sect), config_path, cfg_sect) for cfg_sect in cfg_sects]
prepdict_general = dict(raw_list)


# Get ASN for sequential tests as well as nominal and achieved FDR and FNR
avg_samp_frame_pois_general = pandas.DataFrame(dict([(str(tf), fullrec["ave_samp_pois"]) 
                                                     for tf, fullrec in prepdict_general.items()]))
xxx= pandas.DataFrame(dict([(tf, est_equiv_sample_size(fullrec["config"].getfloat("alpha_general"), 
                      hyp_type="pois", p0=fullrec["config"].getfloat("lam0"), p1=fullrec["config"].getfloat("lam0"),
                      m_null=tf[0], m_alt=tf[1], 
                      base_periods=2, 
                      seqlogfnr_gen=log(fullrec["FDR"]["pois"]["gen_FNR"]), 
                      seqlogfnr_rej=log(fullrec["FDR"]["pois"]["rej_FNR"]), 
                      seqlogfnr_gen_nominal=log(fullrec["config"].getfloat("beta_general")))) for tf, fullrec in 
prepdict_general.items()]))
