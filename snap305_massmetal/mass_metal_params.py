import numpy as np
import sys, os
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy.observate import load_filters
from prospect.models.templates import adjust_dirichlet_agebins
from scipy.stats import truncnorm



#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'debug': False,
              'sed_file': sys.argv[2],
              'outfile': sys.argv[3],
              # dynesty params
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'auto', # sampling method
              'nested_walks': 50, # MC walks
              'nested_nlive_batch': 200, # size of live point "batches"
              'nested_nlive_init': 200, # number of initial live points
              'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
              'nested_dlogz_init': 0.01,
              }

############
# OBS
#############



galex = ['galex_FUV']
hst_wfc3_uv  = ['wfc3_uvis_f275w', 'wfc3_uvis_f336w', 'wfc3_uvis_f475w','wfc3_uvis_f555w', 'wfc3_uvis_f606w', 'wfc3_uvis_f814w']
sdss = ['sdss_i0']
hst_wfc3_ir = ['wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']
spitzer_mips = ['spitzer_mips_24']

herschel_pacs = ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']
herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']

filternames = (galex + hst_wfc3_uv + sdss + hst_wfc3_ir + spitzer_mips + herschel_pacs + herschel_spire)


def load_obs(sed_file = run_params['sed_file'],  **kwargs):

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

    obs = {}
    obs['filters'] = filters
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

##########################
# TRANSFORMATION FUNCTIONS
##########################
def load_gp(**extras):
    return None, None

def tie_gas_logz(logzsol=None, **extras):
    return logzsol

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def massmet_to_logmass(massmet=None,**extras):
    return massmet[0]

def massmet_to_logzsol(massmet=None,**extras):
    return massmet[1]

def logmass_to_masses(massmet=None, logsfr_ratios=None, agebins=None, **extras):
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**massmet[0]) / coeffs.sum()

    return m1 * coeffs


def zfrac_to_masses(massmet=None, z_fraction=None, agebins=None, **extras):
    """This transforms from independent dimensionless `z` variables to sfr
    fractions and then to bin mass fractions. The transformation is such that
    sfr fractions are drawn from a Dirichlet prior.  See Betancourt et al. 2010
    and Leja et al. 2017
    :param total_mass:
        The total mass formed over all bins in the SFH.
    :param z_fraction:
        latent variables drawn form a specific set of Beta distributions. (see
        Betancourt 2010)
    :returns masses:
        The stellar mass formed in each age bin.
    """
    # sfr fractions
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    masses = (10**massmet[0]) * mass_fraction
    return masses



#############
# MODEL_PARAMS
#############
model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.1,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'lumdist', 'N': 1,
                        'isfree': False,
                        'init': 1.0e-5,
                        'units': 'Mpc'})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '', 'prior': None})

model_params.append({'name': 'massmet', 'N': 2,
                        'isfree': True,
                        'init': np.array([10,-0.5]), 'prior': None})

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': False,
                        'depends_on': massmet_to_logmass,
                        'init': 10.0,
                        'units': 'Msun', 'prior': None})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': False,
                        'init': -0.5,
                        'depends_on': massmet_to_logzsol,
                        'units': r'$\log (Z/Z_\odot)$', 'prior': None})
                        
###### SFH   ########
#model_params.append({'name': 'sfh', 'N':1,
#                        'isfree': False,
#                        'init': 0,
#                        'units': None, 'prior': None})
#
#model_params.append({'name': 'mass', 'N': 1,
#                     'isfree': False,
#                     'depends_on': logmass_to_masses,
#                     'init': 1.,
#                     'units': r'M$_\odot$', 'prior': None})
#
#model_params.append({'name': 'agebins', 'N': 1,
#                        'isfree': False,
#                        'init': [],
#                        'units': 'log(yr)', 'prior': None})
#
#model_params.append({'name': 'logsfr_ratios', 'N': 7,
#                        'isfree': True,
#                        'init': [],
#                        'units': '', 'prior': None})




model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 0, "units": "FSPS index"})

model_params.append({'name': "mass", 'N': 3, 'isfree': False, 'init': 1., 'units': r'M$_\odot$',
                          'depends_on': zfrac_to_masses})

model_params.append({'name': "agebins", 'N': 1, 'isfree': False,
                          'init': [],
                          'units': 'log(yr)'})

model_params.append({'name': "z_fraction", "N": 2, 'isfree': True, 'init': [0, 0], 'units': None,
                          'prior': priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=1.0)})







########    IMF  ##############
model_params.append({'name': 'imf_type', 'N': 1,
                             'isfree': False,
                             'init': 2, #Kroupa
                     'prior': ' '
                     })

######## Dust Absorption ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'index', 'prior': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'depends_on': to_dust1,
                        'init': 1.0,
                        'units': '', 'prior': None})

model_params.append({'name': 'dust1_fraction', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.8,
                        'disp_floor': 0.8,
                        'units': '',
                        'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.TopHat(mini=-1.0, maxi=0.4)})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '', 'prior': None})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)', 'prior': None})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None, 'prior': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': 0.2,
                        'disp_floor': 0.15,
                        'units': None,
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 5.0,
                        'disp_floor': 4.5,
                        'units': None,
                        'prior': priors.TopHat(mini=0.1, maxi=25.0)})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': True,
                        'init': 2.0,
                        'init_disp': 3.0,
                        'disp_floor': 3.0,
                        'units': 'percent',
                        'prior': priors.TopHat(mini=0.0, maxi=7.0)})



####### Units ##########
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False, 'prior': None})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mformed', 'prior': None})

#### resort list of parameters for later display purposes
#parnames = [m['name'] for m in model_params]
#fit_order = ['massmet','logsfr_ratios', 'dust2', 'dust_index', 'dust1_fraction', 'duste_gamma', 'duste_umi#n', 'duste_qpah']
#tparams = [model_params[parnames.index(i)] for i in fit_order]
#for param in model_params: 
#    if param['name'] not in fit_order:
#        tparams.append(param)
#model_params = tparams





##### Mass-metallicity prior ######
class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt('/ufrc/narayanan/s.lower/simSEDs/simbam25n512_newfof/snap305_massmetal/gallazzi_05_massmet.txt')

    def __len__(self):
        """ Hack to work with Prospector 0.3
        """
        return 2

    def scale(self,mass):
        upper_84 = np.interp(mass, self.massmet[:,0], self.massmet[:,3]) 
        lower_16 = np.interp(mass, self.massmet[:,0], self.massmet[:,2])
        return (upper_84-lower_16)

    def loc(self,mass):
        return np.interp(mass, self.massmet[:,0], self.massmet[:,1])

    def get_args(self,mass):
        a = (self.params['z_mini'] - self.loc(mass)) / self.scale(mass)
        b = (self.params['z_maxi'] - self.loc(mass)) / self.scale(mass)
        return [a, b]

    @property
    def range(self):
        return ((self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi']))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x:
            x[0] = mass, x[1] = metallicity. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.atleast_2d(np.zeros_like(x))
        a, b = self.get_args(x[...,0])
        p[...,1] = self.distribution.pdf(x[...,1], a, b, loc=self.loc(x[...,0]), scale=self.scale(x[...,0]))
        with np.errstate(invalid='ignore'):
            p[...,1] = np.log(p[...,1])
        return p

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = np.random.uniform(low=self.params['mass_mini'],high=self.params['mass_maxi'],size=nsample)
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), scale=self.scale(mass), size=nsample)

        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = x[0]*(self.params['mass_maxi'] - self.params['mass_mini']) + self.params['mass_mini']
        a, b = self.get_args(mass)
        met = self.distribution.ppf(x[1], a, b, loc=self.loc(mass), scale=self.scale(mass))
        return np.array([mass,met])


def load_sps(**extras):

    sps = FastStepBasis(compute_vega_mags=False, zcontinuous=1, **extras)
    return sps




#print(model_params)
def load_model(nbins_sfh=10, sigma=0.3, df=2, **extras):
    
    # we'll need this to access specific model parameters
    
    n = [p['name'] for p in model_params]
    
    
    agelims_log = np.array([1e-9, 0.1, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 13.6])
    agelims = np.asarray(np.log10(agelims_log) + 9)
    

    agebins = np.array([agelims[:-1], agelims[1:]]).T
    nbins_sfh = len(agelims_log) - 1
    # load nvariables and agebins
    #model_params[n.index('agebins')]['N'] = nbins_sfh
    #model_params[n.index('agebins')]['init'] = agebins
    #model_params[n.index('mass')]['N'] = nbins_sfh
    #model_params[n.index('z_fraction')]['N'] = nbins_sfh-1
    #model_params[n.index('logsfr_ratios')]['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    #model_params[n.index('logsfr_ratios')]['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
    #                                                                  scale=np.full(nbins_sfh-1,sigma),
    #                                                                  df=np.full(nbins_sfh-1,df))
    
    


    ncomp = nbins_sfh
    # constant SFR
    zinit = np.array([(i-1)/float(i) for i in range(ncomp, 1, -1)])

    # Set up the prior in `z` variables that corresponds to a dirichlet in sfr
    # fraction.  THIS IS IMPORTANT
    alpha = np.arange(ncomp-1, 0, -1)
    zprior = priors.Beta(alpha=alpha, beta=np.ones_like(alpha), mini=0.0, maxi=1.0)

    model_params[n.index('mass')]['N'] = ncomp
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins
    model_params[n.index('z_fraction')]['N'] = len(zinit)
    model_params[n.index('z_fraction')]['init'] = zinit
    model_params[n.index('z_fraction')]['prior'] = zprior






    # set mass-metallicity prior
    # insert redshift into model dictionary
    model_params[n.index('massmet')]['prior'] = MassMet(z_mini=-1.98, z_maxi=0.19, mass_mini=7, mass_maxi=13)
    
    #print('\n\nmodel params:', model_params)


    model = sedmodel.SedModel(model_params)

    return model

