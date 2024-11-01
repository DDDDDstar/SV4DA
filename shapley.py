import numpy as np
import random
import math
import time
from privacy_utils import privacy_protect
# from arguments import args_parser
from scipy.optimize import minimize  # , linprog
# from sympy import Symbol, solve, Eq
import datetime
import threading
import queue
import pulp
import warnings
warnings.filterwarnings('ignore')


class Shapley():
    def __init__(self, player_num, task_utility_func, args, targetedplayer_id=None):
        self.args = args

        # self.task = args.task
        self.player_num = player_num
        self.targetedplayer_id = (targetedplayer_id
                                  if targetedplayer_id != None
                                  else range(player_num))
        self.utilityComputation = task_utility_func

        # utility setting
        # self.utility_func = args.utility_func

        # SV settings
        self.SV = dict([(player_id, 0.0)
                        for player_id in range(self.player_num)])
        self.SV_var = dict([(player_id, [])
                            for player_id in range(self.player_num)])
        # SV computation method's components
        self.method = args.method
        self.sampling_strategy = args.sampling_strategy
        self.truncationFlag = args.truncation
        self.privacy_protection_measure = args.privacy_protection_measure
        self.privacy_protection_level = args.privacy_protection_level

        # utility information
        # self.utility_records = dict()
        self.taskTotalUtility = 0
        self.emptySet_utility = 0

        # runtime records
        self.startTime = 0
        self.num_utility_comp = 0
        self.timeCost_per_utility_comp = []

    def truncation(self, truncation, bef_addition):
        if truncation == False:
            return False
        truncation_flag = False
        # print ('check whether to truncate...')
        if np.abs((self.taskTotalUtility - bef_addition) /
                  (self.taskTotalUtility+10**(-15))) < self.args.truncation_threshold:
            truncation_flag = True
            # print("[Truncation!]",
            #      self.taskTotalUtility, bef_addition, self.args.truncation_threshold)
        return truncation_flag

    def generateRandomPermutation(self, ori_permutation, exclude_list=set()):
        permutation = ori_permutation
        while ",".join(map(str, permutation)) in exclude_list:
            random.shuffle(permutation)
        return permutation

    def generateRandomSubset(self, N, q_k, exclude_list=set()):
        k = np.random.choice(range(1, N), p=q_k, size=1)
        selected_players = np.random.choice(
            range(N), int(k), replace=False)
        while ",".join(map(str, sorted(selected_players))) in exclude_list:
            k = np.random.choice(range(1, N), p=q_k, size=1)
            selected_players = np.random.choice(
                range(N), int(k), replace=False)
        return selected_players

    # def sampling(self, sampling_strategy, iter_time,
    #              num_players, scanned_permutations):
    #     permutation = list(range(num_players))
    #     if sampling_strategy == 'antithetic':
    #         if iter_time % 2 == 1:
    #             permutation = self.generateRandomPermutation(
    #                 permutation, scanned_permutations)
    #         else:
    #             # antithetic sampling (also called paired sampling)
    #             permutation = list(reversed(permutation))
    #     elif sampling_strategy == 'stratified':
    #         if iter_time % num_players == 1:
    #             permutation = self.generateRandomPermutation(
    #                 permutation, scanned_permutations)
    #         else:
    #             # stratified sampling
    #             permutation = permutation[-1:] + permutation[:-1]
    #     else:
    #         permutation = self.generateRandomPermutation(
    #             permutation, scanned_permutations)
    #     return permutation

    def sampling(self, sampling_strategy, iter_time,
                 current_permutation, scanned_permutations):
        if sampling_strategy == 'antithetic':
            if iter_time % 2 == 1:
                permutation = self.generateRandomPermutation(
                    current_permutation, scanned_permutations)
            else:
                # antithetic sampling (also called paired sampling)
                permutation = list(reversed(current_permutation))
        elif sampling_strategy == 'stratified':
            if iter_time % len(current_permutation) == 1:
                permutation = self.generateRandomPermutation(
                    current_permutation, scanned_permutations)
            else:
                # stratified sampling
                permutation = current_permutation[-1:] + \
                    current_permutation[:-1]
        else:
            permutation = self.generateRandomPermutation(
                current_permutation, scanned_permutations)
        return permutation

    def PlayerIteration(self, order, player_id, permutation, iter_time,
                        truncation, convergence_diff, diff_mode='relative'):
        startTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # lock.acquire()
        subset = permutation[:order]
        # utility before adding the targeted player
        bef_addition, timeCost = self.utilityComputation(subset)
        if timeCost > 0:
            self.num_utility_comp += 1
            self.timeCost_per_utility_comp.append(timeCost)

        if self.truncation(truncation, bef_addition):
            aft_addition = bef_addition
        else:
            # utility after adding the targeted player
            aft_addition, timeCost = self.utilityComputation(
                list(subset)+[player_id])
            if timeCost > 0:
                self.num_utility_comp += 1
                self.timeCost_per_utility_comp.append(timeCost)
        # update SV
        old_SV = self.SV[player_id]
        # start updating
        self.SV[player_id] = (iter_time-1)/iter_time * old_SV + \
            1/iter_time * (aft_addition - bef_addition)
        self.SV_var[player_id].append(self.SV[player_id])
        # compute difference
        diff = np.abs((self.SV[player_id] - old_SV) /
                      (self.SV[player_id] +
                       (10**(-12) if self.SV[player_id] == 0 else 0)))
        print(('[%s -- %s] Player %s at position %s/%s: utility_bef: %s, ' +
              'utility_aft: %s, SV_bef: %s, SV_aft: %s, diff: %s.') % (
                  startTime, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  player_id, order, len(permutation),
                  bef_addition, aft_addition,
                  old_SV, self.SV[player_id], diff))
        # lock.release()
        if diff_mode == 'utility_diff':
            convergence_diff[player_id] = aft_addition-bef_addition
        else:
            convergence_diff[player_id] = diff

    def MC(self, sampling_strategy='random', truncation=False):

        # Monte Carlo sampling
        self.SV = dict([(player_id, 0.0)
                        for player_id in range(self.player_num)])
        convergence = False
        iter_time = 0
        permutation = list(range(self.player_num))
        scanned_permutations = set()
        convergence_diff_records = list()
        while not convergence:
            iter_time += 1
            permutation = self.sampling(sampling_strategy, iter_time,
                                        permutation, scanned_permutations)
            scanned_permutations.add(",".join(map(str, permutation)))

            # print('\n Monte Carlo iteration %s: ' % iter_time, permutation)
            # speed up by multiple threads
            convergence_diff = dict()
            for order, player_id in enumerate(permutation):
                if player_id not in self.targetedplayer_id:
                    # use self.args.convergence_threshold (instead of 0) for avoiding
                    # too early convergence of computing the targeted player's SV
                    convergence_diff[player_id] = self.args.convergence_threshold
                    continue
                # parallelThreads may lead to the utility of
                # the same player coalition to be computed for two times
                # since the players in the utility_aft computation with
                # permutation[:order]+[player_id] are the same as those
                # in the utility_bef computation with permutation[:order+1]
                thread = threading.Thread(
                    target=self.PlayerIteration,
                    args=(order, player_id, permutation,
                          iter_time, truncation, convergence_diff))
                thread.daemon = True
                thread.start()
                if self.args.num_parallelThreads <= 1 or\
                        (order > 0 and order % self.args.num_parallelThreads == 0):
                    thread.join()
                    print('Done %s/%s...' % (order, len(permutation)))

            while len(convergence_diff) != self.player_num:
                time.sleep(3)
            print('Monte Carlo iteration %s done ' % iter_time)
            convergence_diff_records.append(
                sum(convergence_diff.values())/len(convergence_diff))

            print("Current convergence_diff: ", convergence_diff.values())
            print("Current average convergence_diff: ",
                  convergence_diff_records[-1])
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ",
                  np.average(self.timeCost_per_utility_comp))

            if sampling_strategy == 'stratified' and \
                    iter_time % self.player_num != 0:
                # ensure for each player that the number of permutation
                # samples in each stratum is the same
                continue

            if sampling_strategy == 'antithetic' and \
                    iter_time % 2 != 0:
                # ensure the correctness for paired sampling
                continue

            max_iternum = math.factorial(self.player_num)
            if self.method == 'exact':
                if len(scanned_permutations) >= max_iternum:
                    convergence = True
            elif self.method == 'MC':
                if len(scanned_permutations) >= min(self.args.scannedIter_maxNum, max_iternum):
                    convergence = True
                else:
                    # consider as convergence only when
                    # convergence_diff values in the latest five rounds
                    # are all smaller than the given threshold
                    convergence = True
                    for convergence_diff in convergence_diff_records[-5:]:
                        if convergence_diff > self.args.convergence_threshold:
                            convergence = False
                            break

    def MLE_parallelableThread(self, q, M, sampling_strategy='random',
                               truncation=False, results=None):
        for m in range(M):
            # generate Bernoulli random numbers independently
            I_mq = np.random.binomial(1, q,
                                      size=(self.player_num))
            # print(I_mq)
            subset = []
            for player_id in range(self.player_num):
                if I_mq[player_id] == 1:
                    subset.append(player_id)
            # utility before adding the targeted player
            bef_addition, timeCost = self.utilityComputation(subset)
            results.put((-1, -1, timeCost))
            # self.num_utility_comp += 1
            # self.timeCost_per_utility_comp.append(timeCost)

            for player_id in range(self.player_num):
                if I_mq[player_id] == 1:
                    # needed according to the original paper
                    results.put((player_id, 0, 0))
                    continue
                if self.truncation(truncation, bef_addition):
                    timeCost = 0
                    aft_addition = bef_addition
                else:
                    # utility after adding the targeted player
                    aft_addition, timeCost = self.utilityComputation(
                        list(subset)+[player_id])
                    # self.num_utility_comp += 1
                    # self.timeCost_per_utility_comp.append(timeCost)
                results.put((player_id, aft_addition-bef_addition, timeCost))
                # e[player_id] += aft_addition-bef_addition

            if self.sampling_strategy != 'antithetic':
                continue
            # antithetic sampling
            I_mq = 1-I_mq
            subset = []
            for player_id in range(self.player_num):
                if I_mq[player_id] == 1:
                    subset.append(player_id)
            # utility before adding the targeted player
            bef_addition, timeCost = self.utilityComputation(subset)
            results.put((-1, -1, timeCost))
            # self.num_utility_comp += 1
            # self.timeCost_per_utility_comp.append(timeCost)

            for player_id in range(self.player_num):
                if I_mq[player_id] == 1:
                    # needed according to the original paper
                    results.put((player_id, 0, 0))
                    continue
                if self.truncation(truncation, bef_addition):
                    timeCost = 0
                    aft_addition = bef_addition
                else:
                    # utility after adding the targeted player
                    aft_addition, timeCost = self.utilityComputation(
                        list(subset)+[player_id])
                    # self.num_utility_comp += 1
                    # self.timeCost_per_utility_comp.append(timeCost)
                results.put((player_id, aft_addition-bef_addition, timeCost))
                # e[player_id] += aft_addition-bef_addition

    def MLE(self, sampling_strategy='random', truncation=False):
        # multilinear extension
        # refer to paper: A Multilinear Sampling Algorithm to Estimate Shapley Values
        self.SV = dict([(player_id, 0.0)
                        for player_id in range(self.player_num)])

        convergence_diff_records = []
        convergence = False
        init_MLE_interval = 1000
        MLE_interval = init_MLE_interval
        while not convergence:
            M = 2
            MLE_interval += 2
            # if self.sampling_strategy != 'antithetic':
            if self.sampling_strategy == 'antithetic':
                num_iter = int(MLE_interval/2)+1
            else:
                num_iter = MLE_interval+1

            print(
                'Multilinear extension iteration (with MLE_interval_%s) start!' % MLE_interval)
            threads = []  # speed up by multiple threads
            results = queue.Queue()
            for iter_ in range(num_iter):
                thread = threading.Thread(
                    target=self.MLE_parallelableThread,
                    args=(iter_/MLE_interval, M,
                          sampling_strategy, truncation, results))
                thread.daemon = True
                thread.start()
                if self.args.num_parallelThreads <= 1 or\
                        (iter_ > 0 and iter_ % self.args.num_parallelThreads == 0):
                    thread.join()
                    if self.args.num_parallelThreads > 1 or iter_ % 10 == 0:
                        print('Done %s/%s  (with MLE_interval_%s) ...' % (
                            iter_, num_iter, MLE_interval))
                else:
                    threads.append(thread)
            for thread in threads:
                thread.join()

            e = np.zeros(self.player_num)
            num_comp = np.zeros(self.player_num)
            while not results.empty():
                (player_id, delta_utility, timeCost) = results.get()
                if player_id != -1:
                    e[player_id] += delta_utility
                    num_comp[player_id] += 1
                if timeCost > 0:
                    self.num_utility_comp += 1
                    self.timeCost_per_utility_comp.append(timeCost)

            # convergence check
            convergence_diff = dict([
                (player_id, np.abs(
                    (e[player_id] / num_comp[player_id] - self.SV[player_id]) /
                    (self.SV[player_id] + 10**(-12))
                ))
                for player_id in range(self.player_num)])
            convergence_diff_records.append(
                sum(convergence_diff.values())/len(convergence_diff))
            self.SV = dict([(player_id, e[player_id] / num_comp[player_id])
                            for player_id in range(self.player_num)])
            for player_id in range(self.player_num):
                self.SV_var[player_id].append(self.SV[player_id])
            print(
                'Multilinear extension iteration (with MLE_interval_%s) done ' % MLE_interval)
            print("Current convergence_diff: ", convergence_diff.values())
            print("Current average convergence_diff: ",
                  convergence_diff_records[-1])
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ",
                  np.average(self.timeCost_per_utility_comp))

            if MLE_interval >= self.args.MLE_maxInterval:
                convergence = True
            else:
                # consider as convergence only when
                # convergence_diff values in the latest five rounds
                # are all smaller than the given threshold
                convergence = True
                for convergence_diff in convergence_diff_records[-5:]:
                    if convergence_diff > self.args.convergence_threshold:
                        convergence = False
                        break

    def GT(self, sampling_strategy='random', truncation=False):

        self.SV = dict([(player_id, 0.0)
                        for player_id in range(self.player_num)])
        # group testing
        N = self.player_num
        Z = 2 * sum([1/k for k in range(1, N)])
        q_k = [1/Z*(1/k+1/(N-k)) for k in range(1, N)]

        convergence = False
        iter_time = 0
        scanned_coalitions = set()
        utilities = []
        convergence_diff_records = list()
        while not convergence:
            iter_time += 1

            if sampling_strategy == 'antithetic':
                # antithetic sampling (also called paired sampling)
                if iter_time % 2 == 1:
                    selected_players = self.generateRandomSubset(
                        N, q_k, scanned_coalitions)
                else:
                    selected_players = [player_id
                                        for player_id in range(N)
                                        if player_id not in selected_players]

            elif sampling_strategy == 'stratified':
                pass
                # stratified sampling
                # k = q_k[iter_time%self.player_num-1]
                # selected_players = self.generateRandomSubset(
                #    N, k, scanned_coalitions)

            else:
                selected_players = self.generateRandomSubset(
                    N, q_k, scanned_coalitions)
            scanned_coalitions.add(
                ",".join(map(str, sorted(selected_players))))

            # compute utility
            value, timeCost = self.utilityComputation(selected_players)
            utilities.append(
                ([int(player_id in selected_players)
                  for player_id in range(N)],
                 value)
            )
            if timeCost > 0:
                self.num_utility_comp += 1
                self.timeCost_per_utility_comp.append(timeCost)

            if iter_time % N != 0:
                continue

            print('\n Group testing iteration %s with (k=%s): ' % (
                iter_time, len(selected_players)))
            delta_utility = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    delta_utility[i, j] = Z/iter_time * sum(
                        [utility*(beta[i]-beta[j])
                         for (beta, utility) in utilities])

                    delta_utility[j, i] = - delta_utility[i, j]

            # find SV by solving the feasibility problem
            '''
            # sympy cannot work when inequality has more than one symbol of interest.
            sv = [Symbol('x%s'%player_id) for player_id in range(N)]
            fn = [Eq(sum(sv)-self.taskTotalUtility,0),]
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    ineq = np.abs((sv[i]-sv[j])-delta_utility[i,j])<=self.args.GT_epsilon/2/np.sqrt(N)
                    fn.append(ineq.as_expr())
            result = solve(fn, *sv)
            '''
            MyProbLP = pulp.LpProblem("LPProbDemo1", sense=pulp.LpMaximize)
            sv = [pulp.LpVariable('%s' % player_id, cat='Continuous')
                  for player_id in range(N)]
            MyProbLP += sum(sv)
            # A=[]
            # =[]
            for i in range(N):
                # for j in range(N):
                #    if i==j:
                #        continue
                for j in range(i+1, N):
                    MyProbLP += (sv[i]-sv[j]-delta_utility[i, j]
                                 <= self.args.GT_epsilon/2/np.sqrt(N))
                    MyProbLP += (sv[i]-sv[j]-delta_utility[i, j]
                                 >= -self.args.GT_epsilon/2/np.sqrt(N))
                    # A.append([(1 if k==i else (-1 if k==j else 0)) \
                    #          for k in range(N)])
                    # b.append(self.args.GT_epsilon/2/np.sqrt(N) + delta_utility[i,j])
                    # A.append([(-1 if k==i else (1 if k==j else 0)) \
                    #          for k in range(N)])
                    # b.append(self.args.GT_epsilon/2/np.sqrt(N) - delta_utility[i,j])

            # A = np.array(A) #sparse.csr_matrix(A) #
            # b = np.array(b)
            print("feasible problem solving ...")
            MyProbLP += (sum(sv) >= self.taskTotalUtility)
            MyProbLP += (sum(sv) <= self.taskTotalUtility)
            MyProbLP.solve()
            # status： “Not Solved”, “Infeasible”, “Unbounded”, “Undefined” or “Optimal”
            print("Status:", pulp.LpStatus[MyProbLP.status])
            result = dict()
            for v in MyProbLP.variables():
                # print(v.name, "=", v.varValue)
                result[int(v.name)] = v.varValue
            print('One solution for reference:', v.name, "=", v.varValue)
            print("F(x) = ", pulp.value(MyProbLP.objective),
                  self.taskTotalUtility)  # 输出最优解的目标函数值

            '''
            res = linprog(c=[1 for _ in range(N)],
                          A_ub=A, b_ub=b,
                          A_eq=np.array([[1 for _ in range(N)]]), 
                          b_eq=np.array([self.taskTotalUtility]), 
                          options={'maxiter': 300, 'sparse':True}, 
                          #='revised simplex' # much more accurate method
                          )
            result = res.x
            if res.status == 1:
                print('Iteration limit reached (fun %s)...'%res.fun)
            elif res.status == 2:
                print('Problem appears to be infeasible (fun %s)...'%res.fun)
            elif res.status == 3:
                print('Problem appears to be unbounded (fun %s)...'%res.fun)
            elif res.status == 4:
                print('Numerical difficulties encountered (fun %s)...'%res.fun)
            else:
                print('Problem proceeding nominally (fun %s)...'%res.fun)
            '''
            # convergence check
            convergence_diff = dict([
                (player_id, np.abs(
                    (result[player_id] - self.SV[player_id]) /
                    (self.SV[player_id] + 10**(-12))
                ))
                for player_id in range(self.player_num)])
            convergence_diff_records.append(
                sum(convergence_diff.values())/len(convergence_diff))
            self.SV = dict([(player_id, result[player_id])
                            for player_id in range(self.player_num)])
            for player_id in range(self.player_num):
                self.SV_var[player_id].append(self.SV[player_id])

            print('Group testing iteration %s done!' % iter_time)
            print("Current convergence_diff: ", convergence_diff.values())
            print("Current average convergence_diff: ",
                  convergence_diff_records[-1])
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ",
                  np.average(self.timeCost_per_utility_comp))

            # math.factorial(self.player_num):
            if len(scanned_coalitions) >= 2**self.player_num:
                convergence = True
            else:
                # consider as convergence only when
                # convergence_diff values in the latest five rounds
                # are all smaller than the given threshold
                convergence = True
                for convergence_diff in convergence_diff_records[-5:]:
                    if convergence_diff > self.args.convergence_threshold:
                        convergence = False
                        break

    def CP(self, sampling_strategy='random', truncation=False):

        self.SV = dict([(player_id, 0.0)
                        for player_id in range(self.player_num)])
        # compressive permutation sampling
        # sample a Bernoulli matirc A
        N = self.player_num
        A = np.random.binomial(1, 0.5,
                               size=(self.args.num_measurement, N))
        A = 1/np.sqrt(self.args.num_measurement)*(2*A - 1)
        y = dict([(m, []) for m in range(self.args.num_measurement)])
        # np.zeros(self.args.num_measurement,
        #         math.factorial(self.player_num))

        convergence = False
        iter_time = 0
        permutation = list(range(self.player_num))
        scanned_permutations = set()
        convergence_diff_records = []
        while not convergence:
            iter_time += 1
            permutation = self.sampling(sampling_strategy, iter_time,
                                        permutation, scanned_permutations)
            scanned_permutations.add(",".join(map(str, permutation)))

            print('\n Compressive permutation sampling iteration %s: ' % iter_time,
                  permutation[-5:])
            # speed up by multiple threads
            phi_t = dict()
            for order, player_id in enumerate(permutation):
                thread = threading.Thread(
                    target=self.PlayerIteration,
                    args=(order, player_id, permutation, iter_time, truncation, phi_t, 'utility_diff'))
                thread.daemon = True
                thread.start()
                if self.args.num_parallelThreads <= 1 or\
                        (order > 0 and order % self.args.num_parallelThreads == 0):
                    thread.join()
                    print('Done %s/%s...' % (order, len(permutation)))
            while len(phi_t) != self.player_num:
                time.sleep(3)

            for m in range(self.args.num_measurement):
                y[m].append(sum([
                    A[m, player_id]*phi
                    for (player_id, phi) in phi_t.items()]))
                # y[m, iter_time] = sum([
                #    A[m, player_id]*phi \
#                    for (player_id, phi) in enumerate(phi_t.items())])

            y_mean = np.zeros(self.args.num_measurement)
            for m in range(len(y)):
                y_mean[m] = 1/iter_time * \
                    sum(y[m][:iter_time])  # sum(y[m,:iter_time])
            sv_mean = self.taskTotalUtility/self.player_num
            def fun(sv_variance): return np.linalg.norm(sv_variance, ord=1)

            cons = (
                {'type': 'ineq', 'fun':
                 lambda sv_variance:
                     self.args.CP_epsilon-np.linalg.norm(
                         A.dot(sv_mean+sv_variance)-y_mean, ord=2)},  # self.args.CP_epsilon-x[0]>0 # inequality means that it is to be non-negative
            )
            res = minimize(fun, np.zeros(N),
                           method='SLSQP', constraints=cons)
            sv_variance = res.x

            # convergence check
            convergence_diff = dict([
                (player_id, np.abs(
                    (sv_mean+sv_variance[player_id]-self.SV[player_id]) /
                    (self.SV[player_id] + 10**(-12))
                ))
                for player_id in range(self.player_num)])
            convergence_diff_records.append(
                sum(convergence_diff.values())/len(convergence_diff))
            self.SV = dict([(player_id, sv_mean+sv_variance[player_id])
                            for player_id in range(self.player_num)])

            print('Compressive permutation sampling iteration %s done!' % iter_time)
            print("Current convergence_diff: ", convergence_diff.values())
            print("Current average convergence_diff: ",
                  convergence_diff_records[-1])
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ",
                  np.average(self.timeCost_per_utility_comp))

            if len(scanned_permutations) >= math.factorial(self.player_num):
                convergence = True
            else:
                # consider as convergence only when
                # convergence_diff values in the latest five rounds
                # are all smaller than the given threshold
                convergence = True
                for convergence_diff in convergence_diff_records[-5:]:
                    if convergence_diff > self.args.convergence_threshold:
                        convergence = False
                        break

    def RE_parallelableThread(self, order, selected_players,
                              truncation=False, results=None):
        results.put((order, self.utilityComputation(selected_players)))

    def RE(self, sampling_strategy='random', truncation=False):
        # regression-based
        self.SV = dict([(player_id, 0.0)
                        for player_id in range(self.player_num)])

        d = self.player_num
        A = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i == j:
                    A[i, j] = sum([1/d/(d-k) for k in range(1, d)]) / \
                        sum([1/k/(d-k) for k in range(1, d)])

                else:
                    A[i, j] = 1/d/(d-1) *\
                        sum([(k-1)/(d-k) for k in range(2, d)]) / \
                        sum([1/k/(d-k) for k in range(1, d)])
        z = np.array([0 for _ in range(d)]).reshape(1, -1)
        utilities = {0: self.emptySet_utility}
        scanned_permutations = set()
        convergence = False
        iter_time = 0
        convergence_diff_records = []
        permutation = list(range(self.player_num))
        while not convergence:
            iter_time += 1
            print('\n Regression iteration %s start! ' % iter_time)

            permutation = self.sampling(sampling_strategy, iter_time,
                                        permutation, scanned_permutations)
            scanned_permutations.add(",".join(map(str, permutation)))
            # speed up by multiple threads
            z_i = []
            results = queue.Queue()
            for order, _ in enumerate(permutation):
                z_i.append([int(player_id in permutation[:order+1])
                            for player_id in range(d)])

                thread = threading.Thread(
                    target=self.RE_parallelableThread,
                    args=(order, permutation[:order+1], truncation, results))
                thread.daemon = True
                thread.start()
                if self.args.num_parallelThreads <= 1 or\
                        (order > 0 and order % self.args.num_parallelThreads == 0):
                    thread.join()
                    print('Done %s/%s...' % (order, len(permutation)))
            while results._qsize() != d:
                time.sleep(3)

            while not results.empty():
                order, (value, timeCost) = results.get()
                utilities[len(z)+order] = value
                if timeCost > 0:
                    self.num_utility_comp += 1
                    self.timeCost_per_utility_comp.append(timeCost)

            # regression computation
            z = np.concatenate((z, np.array(z_i)))
            b = np.zeros((d, 1))
            E_Z = 0.5*np.ones((d, 1))
            for (sample_id, z_i) in enumerate(z):
                # A += 1/len(z) * z_i.reshape((-1,1)).dot(z.reshape(1,-1))
                # b += 1/len(z) * (utilities[sample_id]-zero_utility)*z_i.reshape((-1,1))
                b += (z_i.reshape(-1, 1) * utilities[sample_id] -
                      E_Z * self.emptySet_utility) / len(z)
            inv_A = np.linalg.inv(A)
            ones = np.ones((d, 1))
            beta = np.linalg.inv(A).dot(
                b-ones*(
                    (ones.T.dot(inv_A).dot(b)-self.taskTotalUtility+self.emptySet_utility) /
                    ones.T.dot(inv_A).dot(ones)
                )).reshape(-1)

            # convergence check
            convergence_diff = dict([
                (player_id, np.abs(
                    (beta[player_id]-self.SV[player_id]) /
                    (self.SV[player_id] + 10**(-12))
                ))
                for player_id in range(self.player_num)])
            convergence_diff_records.append(
                sum(convergence_diff.values())/len(convergence_diff))
            self.SV = dict([(player_id, beta[player_id])
                            for player_id in range(self.player_num)])
            for player_id in range(self.player_num):
                self.SV_var[player_id].append(self.SV[player_id])

            print('Regression iteration %s done ' % iter_time)
            print("Current convergence_diff: ", convergence_diff.values())
            print("Current average convergence_diff: ",
                  convergence_diff_records[-1])
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ",
                  np.average(self.timeCost_per_utility_comp))

            if len(scanned_permutations) >= math.factorial(self.player_num):
                convergence = True
            else:
                # consider as convergence only when
                # convergence_diff values in the latest five rounds
                # are all smaller than the given threshold
                convergence = True
                for convergence_diff in convergence_diff_records[-5:]:
                    if convergence_diff > self.args.convergence_threshold:
                        convergence = False
                        break

    def computeAllSubsetUtility(self):
        N = self.player_num
        exclude_list = []
        for k in range(2, N+1):
            num_itered_subsets = len(exclude_list)
            finish_thresholdNumber = math.factorial(
                N) // (math.factorial(k) * math.factorial(N - k))
            print('Utility computation for subsets with %s players...' % k)
            while True:
                selected_players = np.random.choice(
                    range(N), int(k), replace=False)
                while ",".join(map(str, sorted(selected_players))) in exclude_list:
                    selected_players = np.random.choice(
                        range(N), int(k), replace=False)
                exclude_list.append(
                    ",".join(map(str, sorted(selected_players))))

                self.utilityComputation(selected_players)
                print('Progress %s/%s with players %s...' % (
                    len(exclude_list)-num_itered_subsets,
                    finish_thresholdNumber, exclude_list[-1]))
                if len(exclude_list) - num_itered_subsets >= finish_thresholdNumber:
                    break

    def problemScale_statistics(self):
        print('【Problem Scale】')
        print('Total number of players: ', self.player_num)
        print('Total number of utility computations for exact computation:',
              float(self.player_num * 2**(self.player_num-1)))
        print('Total number of permutation sampling for exact computation:',
              float(math.factorial(self.player_num)))

    def CalSV(self):
        self.problemScale_statistics()

        # reset rumtime records
        self.num_utility_comp = 0
        self.timeCost_per_utility_comp = []

        # if self.computation_method == 'exact':
        # exact computation can be implemented by
        # setting the convergence conditions in MC
        #    self.Exact()
        # el
        if self.method in ['exact', 'MC']:
            base_compFunc = self.MC
        elif self.method == 'RE':
            base_compFunc = self.RE
        elif self.method == 'MLE':
            base_compFunc = self.MLE
        elif self.method == 'GT':
            base_compFunc = self.GT
        elif self.method == 'CP':
            base_compFunc = self.CP
        else:
            print("Unknown computation method!!")

        if self.truncationFlag or self.method in ['RE', 'GT', 'CP']:
            self.taskTotalUtility, _ = self.utilityComputation(
                range(self.player_num))
            print('The task\'s total utility: ', self.taskTotalUtility)

        if self.method == 'RE':
            self.emptySet_utility, _ = self.utilityComputation([])
            print('The task\'s emptySet utility: ', self.emptySet_utility)

        self.startTime = time.time()
        if self.args.convergence_threshold == 0:
            self.computeAllSubsetUtility()

        base_compFunc(sampling_strategy=self.sampling_strategy,
                      truncation=self.truncationFlag)
        self.SV = privacy_protect(self.privacy_protection_measure,
                                  self.privacy_protection_level,
                                  self.SV, self.SV_var)
        endTime = time.time()
        print("Final Resultant SVs: ", self.SV)
        print("Total runtime: ", endTime-self.startTime)
        print("Total times of utility computation: ", self.num_utility_comp)
        print("Average time cost of a single time of utility computation: ",
              np.average(self.timeCost_per_utility_comp))

        return self.SV, endTime-self.startTime, self.num_utility_comp, np.average(self.timeCost_per_utility_comp)
