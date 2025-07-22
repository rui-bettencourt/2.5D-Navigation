from casadi import *
import casadi as c
import numpy
import unittest
from types import *
from helpers import *
import itertools
import copy

import os
#GlobalOptions.setCatchErrorsPython(False)

solvers= []

if "SKIP_WORHP_TESTS" not in os.environ and has_nlpsol("worhp")  and not args.ignore_memory_heavy:
  solvers.append(("worhp",{"worhp": {"TolOpti":1e-9}},{"codegen": False,"discrete":False}))
  #solvers.append(("worhp",{"TolOpti":1e-20,"TolFeas":1e-20,"UserHM": False}))
  pass

if "SKIP_FATROP_TESTS" not in os.environ and has_nlpsol("fatrop"):
  flags = []
  if os.name != 'nt':
    flags = ["-Wno-strict-prototypes"]
  codegen = {"std": "c99","extralibs": ["fatrop","blasfeo"],"extra_options":flags}
  solvers.append(("fatrop",{"fatrop": {}},{"codegen":codegen,"discrete":False}))

if "SKIP_SLEQP_TESTS" not in os.environ and has_nlpsol("sleqp"):
  solvers.append(("sleqp",{"print_time":False,"sleqp": {"linesearch": "Approx","feas_tol":1e-7,"stat_tol":1e-7,"slack_tol":1e-7, "hess_eval": "Exact"}},{"codegen": False,"discrete":False}))

if "SKIP_ALPAQA_TESTS" not in os.environ and has_nlpsol("alpaqa"):
  solvers.append(("alpaqa",{"print_time":False,"alpaqa": {"alm.tolerance": 1e-10, "alm.dual_tolerance": 1e-10, "alm.penalty_update_factor": 10, "alm.max_iter": 3000, "alm.print_interval": 1, "panoc.max_iter": 500, "panoc.print_interval": 1, "lbfgs.memory": 2}},{"codegen": False,"discrete":False}))

if "SKIP_IPOPT_TESTS" not in os.environ and has_nlpsol("ipopt"):
  codegen = {"extralibs": ["ipopt"], "std": "c99"}
  solvers.append(("ipopt",{"print_time":False,"ipopt": {"tol": 1e-10, "derivative_test":"second-order","print_level":0}},{"codegen": codegen,"discrete":False}))
  solvers.append(("ipopt",{"print_time":False,"ipopt": {"tol": 1e-10, "derivative_test":"first-order","hessian_approximation": "limited-memory","print_level":0}},{"codegen": codegen,"discrete":False}))

if "SKIP_IPOPT_TESTS" not in os.environ and has_nlpsol("ipopt") and has_nlpsol("sqpmethod"):
  qpsol_options = {"nlpsol": "ipopt", "nlpsol_options": {"ipopt.tol": 1e-12,"ipopt.tiny_step_tol": 1e-20, "ipopt.fixed_variable_treatment":"make_constraint","ipopt.print_level":0,"print_time":False,"print_time":False} }
  solvers.append(("sqpmethod",{"qpsol": "nlpsol","qpsol_options": qpsol_options,"print_header":False,"print_iteration":True,"print_time":False},{"codegen": False,"discrete":False}))
  solvers.append(("sqpmethod",{"qpsol": "nlpsol","qpsol_options": qpsol_options,"hessian_approximation": "limited-memory","tol_du":1e-10,"tol_pr":1e-10,"min_step_size":1e-14,"print_header":False,"print_iteration":True,"print_time":False,"max_iter":55},{"codegen": False,"discrete":False}))

if "SKIP_QRQP_TESTS" not in os.environ and has_conic("qrqp") and has_nlpsol("sqpmethod"):
  codegen = {"std": "c99"}
  qpsol_options = {"print_iter":False,"print_header":False,"error_on_fail" : False}
  solvers.append(("sqpmethod",{"qpsol": "qrqp","qpsol_options": qpsol_options,"print_header":False,"print_iteration":False,"print_time":False},{"codegen":codegen,"discrete":False}))
  solvers.append(("sqpmethod",{"qpsol": "qrqp","max_iter_ls":0,"qpsol_options": qpsol_options,"print_header":False,"print_iteration":False,"print_time":False},{"codegen":codegen,"discrete":False}))
  solvers.append(("sqpmethod",{"qpsol": "qrqp","convexify_strategy":"regularize","max_iter":500,"qpsol_options": qpsol_options,"print_header":False,"print_iteration":True,"print_time":False,"tol_du":1e-8,"min_step_size":1e-12},{"codegen":codegen,"discrete":False}))

if "SKIP_DAQP_TESTS" not in os.environ and has_conic("daqp") and has_nlpsol("sqpmethod"):
  codegen = {"std": "c99","extralibs": ["daqp"]}
  solvers.append(("sqpmethod",{"qpsol": "daqp","hessian_approximation":"limited-memory","tol_du":1e-10,"tol_pr":1e-10,"min_step_size":1e-14,"max_iter":55,"print_header":False,"print_iteration":True,"print_time":False,"qpsol_options":{"print_problem":True,"print_out":True}},{"codegen":codegen,"discrete":False}))

if "SKIP_BLOCKSQP_TESTS" not in os.environ and has_nlpsol("blocksqp"):
  try:
    load_linsol("ma27")
    solvers.append(("blocksqp",{},{"codegen": False,"discrete":False}))
  except:
    pass

if "SKIP_BONMIN_TESTS" not in os.environ and has_nlpsol("bonmin"):
  solvers.append(("bonmin",{},{"discrete":True,"codegen":False}))

if "SKIP_KNITRO_TESTS" not in os.environ and has_nlpsol("knitro"):
  solvers.append(("knitro",{"knitro":{"feastol":1e-9,"opttol":1e-9,"ftol":1e-20}},{"codegen": False,"discrete":False}))

if "SKIP_SNOPT_TESTS" not in os.environ and has_nlpsol("snopt"):
  solvers.append(("snopt",{"snopt": {"Verify_level": 3,"Major_optimality_tolerance":1e-12,"Minor_feasibility_tolerance":1e-12,"Major_feasibility_tolerance":1e-12}},{"codegen": False,"discrete":False}))
  
print(solvers)