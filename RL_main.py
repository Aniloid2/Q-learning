import scipy.io as sio
import numpy as np
import sys
import time
import matplotlib.pyplot as plt


class movement():
	def __init__(self,a,s_prime,k,s, arrow_a):
		self.a = a
		self.s_prime = s_prime
		self.k = k
		self.s = s
		self.arrow_a = arrow_a


class policy(): # no mutation hill-climber
	def __init__(self,best,best_reward, best_raw):
		self.best = best
		self.best_reward = best_reward
		self.best_policy_raw = best_raw

	def update_best(self, competitor, competitor_reward,competitor_raw):
		if competitor_reward >= self.best_reward:
			self.best = Optimal_policy(np.zeros((100),dtype='str'))
			self.best_policy_raw = np.zeros((100,4),dtype='float')
			self.best.initialize()
			self.best = competitor
			self.best_policy_raw = competitor_raw
			self.best_reward = competitor_reward

class Optimal_policy():
	def __init__(self,policy_op):
		self.policy_op = policy_op
		self.movement_weight = np.zeros((100))
		self.policy_vector = np.zeros((100), dtype = 'float')

	def initialize(self):
		for i in range(len(self.policy_op)):
			self.policy_op[i] = ' '
		self.policy_op[-1] = 'x' 

	def movements(self,movements_list):
		for m in movements_list:
			self.movement_weight[m.s] += 1 
			self.policy_op[m.s] = m.arrow_a


class Q_learning():
	"""docstring for Q_learning"""
	def __init__(self):
		self.Qf = np.zeros((100,4),dtype='float')
		self.QPrev = np.zeros((100,4),dtype='float')

		self.s = 0
		self.y = 0.5
		self.T = sio.loadmat('qeval.mat')['reward']


	def get_a_index(self,s,E):
		u = np.random.uniform(0,1) # check between which values
		s_a_options = self.Qf[s]
		alternative = []
		main_options = []
		max_option = max(s_a_options)
		for i in range(len(s_a_options)):
			if s_a_options[i] == max_option:
				main_options.append(i+1)
			else:
				alternative.append(i+1)

		if u <= 1 - E:
			a = np.random.choice(main_options)
		elif u > 1 - E:
			if len(alternative) == 0:
				a = np.random.choice(main_options)
			else:

				a = np.random.randint(1,5)
		return a


	def get_a(self,s,E):
		u = np.random.uniform(0,1) # check between which values
		s_a_options = self.Qf[s]
		alternative = []
		main_options = []
		max_option = max(s_a_options)
		for i in range(len(s_a_options)):
			if s_a_options[i] == max_option:
				main_options.append(max_option)
			else:
				alternative.append(s_a_options[i])
		if u <= 1 - E:
			a = np.random.choice(main_options)
		elif u > 1 - E:
			if len(alternative) == 0:
				a = np.random.choice(main_options)
			else:
				a = np.random.choice(alternative)
		return a



	def get_s_prime(self, s,a):
		if a == 1:
			s_prime = s - 1
		elif a == 2:
			s_prime = s + 10
		elif a == 3:
			s_prime = s + 1
		elif a == 4:
			s_prime = s - 10
		return s_prime

	def get_optimal_policy(self,E):
		Q_star = []
		for s in range(len(self.Qf)):
			a = self.get_a_index(s,E = 0)
			Q_star.append(self.change_a_to_arrow(a))

		return (Q_star)


	def change_a_to_arrow(self,a):
		if a == 1:
			return '^'
		elif a ==2:
			return '>'
		elif a==3:
			return 'v'
		elif a ==4:
			return '<'



	def plot_on_mpl(self,plot_this,r,a,time,trials,best_raw):
		fig, ax = plt.subplots()
		plot_this_resh = plot_this.policy_op.reshape((10,10), order = 'F')
		movs = plot_this.movement_weight.reshape((10,10), order = 'F')
		movs[-1,-1] +=1
		x,y = plot_this_resh.shape

		for i in range(x):
			for j in range(y):
				if (i == 0 and j ==0):
					c= 'r'
					alp = 1
				else:
					c = 'b'
					alp = 0.1


				if movs[i,j] == 1:
					c = '#8080ff'
					alp = 1 
				elif movs[i,j] > 10 and movs[i,j] < 100:
					c = '#0000ff'
					alp = 1
				elif movs[i,j] > 100:
					c = '#000066'
					alp = 1
				if (i == 9 and j ==9):
					c = 'r'
					alp = 1
				if (i == 0 and j ==0):
					c= 'r'
					alp = 1


				ax.plot(j,-i, marker=plot_this_resh[i,j], c=c,  alpha=alp, markersize=10)

		ax.set_title('Path for Alpha, Epsilon: %s Discount: %.01f \n Reward: %.02f Time: %.04f Trials %d/10 ' % (a, self.y , r , time,trials  )) 
		plt.xticks([])
		plt.yticks([])
		plt.savefig('qevalstates.png')
		f = open('qevalstates.txt','w')
		f.write('| a1 | a2 | a3 | a4 | \n')
		for i in range(len(best_raw)):
			wthis = 's: ' + str(i) + ' ' + str(best_raw[i]) + '\n'
			f.write(wthis)
		f.close()


	def trial(self,times):

		self.s = 0
		count =0
		Policy_competition = policy(Optimal_policy(np.zeros((100),dtype='str')),0,np.zeros((100,4),dtype='float'))
		stop = False
		while stop == False:
			average_time = []
			for t in range(times):
				Rt = 0
				start_time = time.time()
				
				movements_list = []
				for k in range(3000):
					# fails all the time, the learning rate and e greedy degrates too quickly to get to the final point
					# Alpha, Alpha_string = 1/(k+1), '1/(k)'
					# E = 1/(k+1)


					# with 0.5 it failed 3 times before outputting a result
					# with 0.9 it failed 3 times before outputting a result
					# Alpha ,  Alpha_string = (1+5*np.log((k+1)))/(k+1), '(1+5*log(k))/k'
					# E = (1+5*np.log((k+1)))/(k+1)



					Alpha, Alpha_string = (100)/(k+101), '(100)/(k+100)'
					E = (100)/(k+101)


					#0.5 4 failed before getting results

					# Alpha , Alpha_string = (1+np.log((k+1)))/(k+1), '(1+log(k))/k'
					# E = (1+np.log((k+1)))/(k+1)

					p = -1
					while p <= -1:
						a = self.get_a_index(self.s, E)
						p = self.T[self.s,a-1]
						print (p)

					Rt += p * self.y**(k)

					s_prime = self.get_s_prime(self.s,a)
					m = movement(a,s_prime,k,self.s, self.change_a_to_arrow(a))
					movements_list.append(m)


					print ( self.s,'<-- ','current_state', s_prime, '<-- new state || ', k)



					print ('reward', p)




					a_prob = self.get_a(s_prime,E)

					print ('probability', a_prob)



					self.Qf[self.s,a-1] = self.QPrev[self.s,a-1] + Alpha*( p + self.y*a_prob - self.QPrev[self.s,a-1])

					self.QPrev[self.s,a-1] = self.Qf[self.s,a-1]

					self.s = s_prime
					end_time = time.time()
					abs_time = end_time - start_time
					if s_prime == 99:
						self.s = 0
						count +=1
						average_time.append(abs_time)
						break
				print (self.Qf,'<--Q_table')

				
				Q_star = np.array(self.get_optimal_policy(E))
				print (Q_star.reshape((10,10),order='F'), '<--Q_star Reward -->',Rt,'Evals', t, 'Time',abs_time)

				

				self.s = 0



				Q_zero = Optimal_policy(np.zeros((100), dtype='str'))
				Q_zero.initialize()
				Q_zero.movements(movements_list)


				print (Q_zero.policy_op.reshape((10,10),order='F'), '<-- Q_zero anser')
				print (Q_zero.movement_weight.reshape((10,10),order='F'), '<-- movements weights')
				if s_prime == 99:
					Policy_competition.update_best(Q_zero,Rt,self.Qf)

				

				try:
					mean_time = sum(average_time)/len(average_time)
				except:
					mean_time = 0

			print (Policy_competition.best.policy_op.reshape((10,10), order='F'),'<-- Best one we can offer', Policy_competition.best_reward)
			print (Policy_competition.best_policy_raw, '<-- best policy raw values')
			self.plot_on_mpl(Policy_competition.best, Policy_competition.best_reward, Alpha_string, mean_time, count, Policy_competition.best_policy_raw )

			print (average_time, mean_time)
			print ('final success', count,'out of', times)
			if Policy_competition.best_reward > 0:
				stop = True





env = Q_learning()
env.trial(10)

