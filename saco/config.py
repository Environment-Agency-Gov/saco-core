import numpy as np


class Constants:
    """
    Global constants for reuse across modules.

    """
    waterbody_id_column = 'EA_WB_ID'

    ra_abb = 'RA'
    fl_abb = 'FL'
    fp_abb = 'FP'
    valid_scenarios = [ra_abb, fl_abb, fp_abb]

    valid_percentiles = [95, 70, 50, 30]

    sub_abb = 'sub'
    ups_abb = 'ups'
    valid_value_types = [sub_abb, ups_abb]

    valid_factor_names = ['scenario', 'percentile', 'value_type']

    qnat_abb = 'QN'
    gwabs_abb = 'GW'
    swabs_abb = 'SW'
    dis_abb = 'DISCH'
    sup_abb = 'SUP'
    efi_abb = 'EFI'
    scen_abb = 'SCEN'
    sd_abb = 'SD'
    comp_abb = 'COMP'
    qt_abb = 'QT'
    sdt_abb = 'SDT'
    asb_abb = 'ASB'

    # Compliance bin edges
    # - gives lower and upper (factor) bounds for each classification (compliant and
    #    band 1/2/3)
    compliance_bin_edges = [-np.inf, -0.5, -0.25, 0.0, np.inf]

    # Optimiser
    arc_index_column = 'ARC_IDX'
    optimise_flag_column = 'Optimise_Flag'
