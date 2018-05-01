import numpy as np

class DDPG:
    def __init__(self, sess, buffer, env, actor, critic, actor_noise, r_mon):
        self.sess = sess
        self.buffer = buffer
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.r_mon = r_mon

    def fill_buffer(self, num_samples, action_range):

        # same as training loop without training
        s = self.env.reset()
        for _ in range(num_samples):
            a = np.random.uniform(-action_range, action_range, size=1)
            s_next, r, d = self.env.step(a)
            self.buffer.add(s, a, r, d, s_next)
            s = s_next
            if d:
                s = self.env.reset()

        print("Done with dummy filling")

    def train(self, gamma, max_episodes, max_episode_len, replay_len):

        # initialize target networks
        self.actor.copy_vars()
        self.critic.copy_vars()
        total_r = 0

        for i in range(max_episodes):

            # reset if terminated
            s = self.env.reset()
            self.actor_noise.reset()

            # write to monitor
            self.r_mon.record(value=total_r, step=i)
            total_r = 0

            for j in range(max_episode_len):

                # predict action
                a = self.actor.predict([s])[0] + self.actor_noise()

                # take action
                s_next, r, d = self.env.step(a)
                total_r += r

                # render
                #if i > 2000:
                #    self.env.env.render(mode='human')

                # add sample to buffer
                self.buffer.add(s, a, r, d, s_next)

                if j % replay_len == 0:

                    # sample from buffer
                    s_batch, a_batch, r_batch, d_batch, s_next_batch = self.buffer.sample_batch()

                    # predict all q_next
                    q_next_batch = self.critic.predict(s_next_batch, self.actor.predict_target(s_next_batch))

                    # create a mask wrt terminal states
                    mask = (np.logical_not(d_batch)).astype(np.float32)

                    # create target values
                    y_batch = r_batch + gamma * q_next_batch * mask

                    # train the critic
                    self.critic.train(s_batch, a_batch, q_target=y_batch)

                    # get new batch of actions
                    new_actions = self.actor.predict(s_batch)

                    # update the actor
                    actions_grads = self.critic.get_gradients(s_batch, new_actions)
                    self.actor.train(s_batch, actions_grads)

                    # update target networks
                    self.actor.update_vars()
                    self.critic.update_vars()

                s = s_next
                if d:
                    break

            print("ep_:" + str(i) + "   r_:" + str(total_r))


        print("Done with training")


