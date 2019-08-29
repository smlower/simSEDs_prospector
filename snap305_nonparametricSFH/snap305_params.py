
# coding: utf-8

# ## Attempt at param script for prospector using snap305 (z = 0) of simba m25_n512 run

# In[1]:


import numpy as np
from prospect.models import priors, SedModel
from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins, adjust_dirichlet_agebins
from prospect.sources import FastStepBasis, CSPSpecBasis
from sedpy.observate import load_filters
import sys, os

# In[2]:


# --------------
# RUN_PARAMS
# --------------

run_params = {'verbose': False,
              'sed_file': str(sys.argv[2]),
              'outfile': str(sys.argv[3]),
              'output_pickles': False,
              #dynesty stuff
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'auto', # sampling method
              'nested_walks': 50, # MC walks
              'nested_nlive_batch': 500, # size of live point "batches"
              'nested_nlive_init': 500, # number of initial live points
              'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
              'nested_dlogz_init': 0.01,
              # Obs data parameters
              'objid':0,
              'logify_spectrum':False,
              'normalize_spectrum':False,
              'luminosity_distance': 1e-5,  # in Mpc
              # Model parameters
              'add_neb': False,
              'add_dust': True,
              # SPS parameters
              'zcontinuous': 1,
              }


# In[7]:


# --------------
# OBS
# --------------

# Here we are going to put together some filter names
galex = ['galex_FUV']
hst_wfc3_uv  = ['wfc3_uvis_f275w', 'wfc3_uvis_f336w', 'wfc3_uvis_f475w','wfc3_uvis_f555w', 'wfc3_uvis_f606w', 'wfc3_uvis_f814w']
sdss = ['sdss_i0']
hst_wfc3_ir = ['wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']
spitzer_mips = ['spitzer_mips_24']
#wise = ['wise_w4']
herschel_pacs = ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']
herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']




# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
filternames = (galex + hst_wfc3_uv + sdss + hst_wfc3_ir + spitzer_mips + herschel_pacs + herschel_spire)



def load_obs(sed_file = run_params['sed_file'], **kwargs):
    
    print('loading galaxy:',os.path.basename(sed_file).split('.')[1])

    from hyperion.model import ModelOutput
    from astropy.cosmology import Planck15
    from astropy import units as u
    from astropy import constants
    m = ModelOutput(sed_file)

    wav,flux = m.get_sed(inclination='all',aperture=-1)
    wav  = np.asarray(wav)*u.micron #wav is in micron
    wav = wav.to(u.AA)
    #wav *= (1.+z)
    flux = np.asarray(flux)*u.erg/u.s
    dl = 10.0*u.pc
    #dl = Planck15.luminosity_distance(z)
    dl = dl.to(u.cm)
    flux /= (4.*3.14*dl**2.)
    nu = constants.c.cgs/(wav.to(u.cm))
    nu = nu.to(u.Hz)

    flux /= nu
    flux = flux.to(u.Jy)
    maggies = flux[0] / 3631.


    filters = load_filters(filternames)
    filter_wavs = [x.wave_mean for x in filters]


    flx = []
    flxe = []
    for i in range(len(filter_wavs)):
        flx.append(maggies[(np.abs(wav.value - filter_wavs[i])).argmin()].value)
        flxe.append(0.01* flx[i])
    flx = np.asarray(flx)
    flxe = np.asarray(flxe)


    flux_mag = flx
    unc_mag = flxe

    # Build output dictionary.                                                                   
    obs = {}
    # This is a list of sedpy filter objects.    See the                                         
    # sedpy.observate.load_filters command for more details on its syntax.                       
    obs['filters'] = load_filters(filternames)
    # This is a list of maggies, converted from mJy.  It should have the same                    
    # order as `filters` above.                                                                  
    obs['maggies'] = flux_mag
    #Uncertainties also converted from mJy. In same order as flux_mag and filters                
    obs['maggies_unc'] = unc_mag
    # Here we mask out any NaNs or infs                                                          
    obs['phot_mask'] = np.isfinite(flux_mag)
    # We have no spectrum.                                                                       
    obs['wavelength'] = None
    

    return obs


# In[5]:


# --------------
# SPS Object
# --------------


"""
This function loads an sps object which is then used to build SEDs for a given set of parameters, powered by FSPS. 
The SED is generated in the next function through SedModel
"""

def load_sps(zcontinuous=1, **extras):
    sps = FastStepBasis(compute_vega_mags=False, zcontinuous=zcontinuous)
    return sps


# In[8]:


# -----------------
# Gaussian Process
# ------------------

"""
Not sure at all what this does. All i know is that prospector cannot run without"""

def load_gp(**extras):
    return None, None

# --------------
# MODEL_PARAMS
# --------------

def load_model(object_redshift=None, fixed_metallicity=None, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object, which generates a spectrum using the previous function's sps object

    """ 
    model_params = TemplateLibrary["dirichlet_sfh"]
    model_params.update(TemplateLibrary["dust_emission"])
    
    model_params["lumdist"] = {"N": 1, "isfree": False,
                                   "init": 1.0e-5, "units":"Mpc"}
    #nbins = 10
    #tuniv = 13.7
    #tbinmax = (tuniv * 0.85) 
    #lim1, lim2 = 7.4772, 8.0
    alpha_agelims = np.array([1e-9, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 13.6])
    model_params = adjust_dirichlet_agebins(model_params, agelims=(np.log10(alpha_agelims) + 9))
    


    model_params['logzsol']['init'] = 0.0
    model_params['logzsol']['prior'] = priors.TopHat(mini=-1.0, maxi=0.5)
    model_params['mass']['prior'] = priors.TopHat(mini=1.e7, maxi=1.e13)
    # Set the dust and agn emission free
    model_params["duste_qpah"]["isfree"] = True
    model_params["duste_umin"]["isfree"] = True
    model_params["duste_gamma"]["isfree"] = True
    model_params['duste_gamma']['init'] = 0.1
    model_params['duste_umin']['prior'] = priors.TopHat(mini=0.1, maxi=35)
    model_params['duste_qpah']['prior'] = priors.TopHat(mini=0.01, maxi=8.0)
    model_params["duste_gamma"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)

        


    
    # Complexify the dust attenuation



    model_params['dust_type']['init']= 4
    model_params['dust1'] = {'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.8,
                        'disp_floor': 0.8,
                        'units': '',
                        'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}
    model_params['dust2'] = {'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)}
    model_params['dust_index'] = {'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.TopHat(mini=-1.0, maxi=0.4)}

    model_params['dust1_index'] = {'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '', 'prior': None}
    model_params['dust_tesc'] = {'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)', 'prior': None}


    # Now instantiate the model using this new dictionary of parameter specifications
    model = SedModel(model_params)

    return model
