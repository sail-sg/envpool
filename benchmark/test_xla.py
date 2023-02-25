import envpool
import jax

env = envpool.make(
    "Pong-v5",
    env_type="dm",
    num_envs=2,
)
handle, recv, send, _ = env.xla()

def actor_step(iter, loop_var):
    handle0, states = loop_var
    action = 0
    handle1 = send(handle0, action, states.observation.env_id)
    handle1, new_states = recv(handle0)
    return handle1, new_states

@jax.jit
def run_actor_loop(num_steps, init_var):
    return jax.lax.fori_loop(0, num_steps, actor_step, init_var)

env.async_reset()
handle, states = recv(handle)
run_actor_loop(100, (handle, states))
