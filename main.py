import params
import funcs

Parameters = params.get_params()
Results    = funcs.simulate(Parameters)
_          = funcs.plot(Results,Parameters)
