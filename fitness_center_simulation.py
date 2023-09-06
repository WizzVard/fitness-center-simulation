import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the gym simulation

MEAN_ARRIVAL_TIME = 3

MEAN_TREADMILL_TIME = 15
MEAN_BARBELL_TIME = 15
MEAN_DUMBBELL_TIME = 15
MEAN_TIME_LEG_PRESS = 15
MEAN_TIME_FITNESS_BALL = 5
MEAN_TIME_STRETCHING = 15
MEAN_TIME_KETTLEBELL = 10
MEAN_TIME_PULL_UP_FRAME = 5
MEAN_TIME_BICYCLE = 15
MEAN_TIME_LEG_EXTENSION_MACHINES_MACHINE = 10
MEAN_TIME_CABLE_MACHINE = 10
MEAN_TIME_LEVER_MACHINE = 10

NUM_PERSONAL_COACHES = 5
GROUP_COACH_CAPACITY = 5

NUM_TREADMILLS = 2
NUM_STATIONARY_BICYCLES = 2
NUM_STRETCHING_SETS = 5
NUM_FITNESS_BALLS = 5

NUM_DUMBBELL_SETS = 1
NUM_PULL_UP_FRAMES = 1
NUM_BARBELL_SETS = 3
NUM_KETTLEBELL_SETS = 1
NUM_CABLE_MACHINES = 1
NUM_LEVER_MACHINES = 1

NUM_LEG_PRESSES = 1
NUM_LEG_EXTENSION_MACHINES = 1

OBSTRUCT_TRAINER_TIME = 10*60
OBSTRUCT_TRAINER_FREQUENCY = 14*60

MEAN_PERSONAL_COACH_TIME = 30
MEAN_GROUP_COACH_TIME = 45

PROB_FOR_TRAINER = 0.6
PROB_FOR_PERSONAL_TRAINER_PLAN = 0.7

BASIC_PACK_PER_DAY = 80/30
VIP_PACK_PER_DAY = 120/30
PERSONAL_TRAINER_PER_DAY = 250/30
GROUP_TRAINER_PER_DAY = 100/30

PROB_FOR_LEG_DAY = 0.3
PROB_FOR_BODY_DAY = 0.5
PROB_FOR_CARDIO_DAY = 0.2

MAX_CLIENTS_AT_ONE_TIME = 30
MAX_CLIENTS = 100_000

SIM_TIME = 24 * 60

NUM_RUNS = 100

total_runs_results = {}
total_customers_handled = []

# Create a DataFrame to store income results
income_df = pd.DataFrame({"Run number": [],
                          "Income from standard clients with personal trainer": [],
                          "Income from standard clients with group trainer": [],
                          "Income from standard clients without trainer": [],
                          "Income from VIP clients with personal trainer": [],
                          "Income from VIP clients with group trainer": [],
                          "Income from VIP clients without trainer": []})
income_df.set_index("Run number", inplace=True)


# Function to simulate using gym equipment
def engage_equipment(equipment, mean_time, equipment_name):
    start_time_minutes = env.now
    # Request access to the equipment
    with equipment.request() as req:
        yield req
        end_time_minutes = env.now

        total_time_here_in_minutes = end_time_minutes - start_time_minutes

        if equipment_name not in run_results:
            run_results[equipment_name] = []

        run_results[equipment_name].append(total_time_here_in_minutes)

        # Simulate using the equipment for a random amount of time
        random_normal = np.random.normal(mean_time, 1)
        yield env.timeout(random_normal)


# Function to simulate a client requesting a trainer
def trainer_request(trainer, client_priority, mean_time, trainer_type):
    start_time_minutes = env.now
    # Request a trainer with a specified priority
    with trainer.request(priority=client_priority) as req:
        yield req
        end_time_minutes = env.now

        total_time_here_in_minutes = end_time_minutes - start_time_minutes

        if trainer_type not in run_results:
            run_results[trainer_type] = []

        run_results[trainer_type].append(total_time_here_in_minutes)

        # Simulate trainer interaction for a random amount of time
        random_normal = np.random.normal(mean_time, 1)
        yield env.timeout(random_normal)


# Function to decide whether a client will take a trainer
def trainer_decision(take_a_trainer):
    # Randomly decide whether a client will take a trainer
    if random.random() <= PROB_FOR_TRAINER:
        take_a_trainer = True

    return take_a_trainer


# Function to decide the daily workout plan
def plan_decision(day_plan):
    # Randomly decide the client's workout plan for the day
    if random.random() <= PROB_FOR_BODY_DAY:
        day_plan = "body_day"

    elif random.random() <= PROB_FOR_BODY_DAY + PROB_FOR_CARDIO_DAY:
        day_plan = "cardio_day"

    return day_plan


# Function to obstruct trainers at specified intervals
def obstruct_trainer(trainer, obstruct_time, obstruct_frequency, trainer_type):
    while True:
        print(f"{trainer_type} will be obstructed at {round(env.now + obstruct_frequency, 2)}")
        # Freeze the function for the unavailability frequency period.
        yield env.timeout(obstruct_frequency)

        # Once this time has elapsed, request a trainer with a priority of -1
        # and hold them for the specified unavailability amount of time
        with trainer.request(priority=-1) as req:
            # Freeze the function until the request can be met
            yield req

            print(f"{trainer_type} is unavailable. Will come bake in {round(env.now + obstruct_time, 2)}")
            yield env.timeout(obstruct_time)


# Function to simulate a client's gym visit
def attend_gym(take_a_trainer, day_plan, client_priority):
    # Generate client decision about trainer and workout plan
    global st_client_with_personal_t_counter
    global st_client_with_group_t_counter
    global vip_client_with_personal_t_counter
    global vip_client_with_group_t_counter
    global st_client_without_t_counter
    global vip_client_without_t_counter

    global run_customers_handled

    trainer = trainer_decision(take_a_trainer)
    day_plan = plan_decision(day_plan)

    if trainer:
        if random.random() <= PROB_FOR_PERSONAL_TRAINER_PLAN:
            # Request a personal trainer if chosen
            yield from trainer_request(personal_coach, client_priority, MEAN_PERSONAL_COACH_TIME, "Personal coach")
            # count client with priorities to compute income
            if client_priority == 1:
                vip_client_with_personal_t_counter += 1

            else:
                st_client_with_personal_t_counter += 1

        else:
            # Request a group coach if chosen
            yield from trainer_request(group_coach, client_priority, MEAN_GROUP_COACH_TIME, "Group coach")
            # count client with priorities to compute income
            if client_priority == 1:
                vip_client_with_group_t_counter += 1

            else:
                st_client_with_group_t_counter += 1
    else:
        # Simulate using gym equipment based on the workout plan
        if day_plan == "cardio_day":
            yield from engage_equipment(treadmill, MEAN_TREADMILL_TIME, "Treadmill")
            yield from engage_equipment(stationary_bicycle, MEAN_TIME_BICYCLE, "Bicycle")
            yield from engage_equipment(stretching_set, MEAN_TIME_STRETCHING, "Stretching set")
            yield from engage_equipment(fitness_ball, MEAN_TIME_FITNESS_BALL, "Fitness ball")

        elif day_plan == "body_day":
            yield from engage_equipment(stretching_set, MEAN_TIME_STRETCHING, "Stretching set")
            yield from engage_equipment(barbell_set, MEAN_BARBELL_TIME, "Barbell set")
            yield from engage_equipment(dumbbell_set, MEAN_DUMBBELL_TIME, "Dumbbell set")
            yield from engage_equipment(pull_up_frame, MEAN_TIME_PULL_UP_FRAME, "Pull up frame")
            yield from engage_equipment(cable_machine, MEAN_TIME_CABLE_MACHINE, "Cable machine")
            yield from engage_equipment(lever_machine, MEAN_TIME_LEVER_MACHINE, "Lever machine")

        else:
            yield from engage_equipment(treadmill, MEAN_TREADMILL_TIME, "Treadmill")
            yield from engage_equipment(barbell_set, MEAN_BARBELL_TIME, "Barbell set")
            yield from engage_equipment(leg_extension_machine, MEAN_TIME_LEG_EXTENSION_MACHINES_MACHINE,
                                        "Leg extension machine")
            yield from engage_equipment(leg_press, MEAN_TIME_LEG_PRESS, "Leg press")
            yield from engage_equipment(kettlebell_set, MEAN_TIME_KETTLEBELL, "Kettlebell set")
            yield from engage_equipment(dumbbell_set, MEAN_DUMBBELL_TIME, "Dumbbell set")

        # count client with priorities to compute income
        if client_priority == 1:
            vip_client_without_t_counter += 1

        else:
            st_client_without_t_counter += 1

    run_customers_handled += 1

# Main simulation loop
for run_number in range(NUM_RUNS):
    print(f"{run_number=} out of {NUM_RUNS}")

    env = simpy.Environment()

    gym_capacity = simpy.PriorityResource(env, capacity=MAX_CLIENTS_AT_ONE_TIME)

    personal_coach = simpy.PriorityResource(env, capacity=NUM_PERSONAL_COACHES)
    group_coach = simpy.PriorityResource(env, capacity=GROUP_COACH_CAPACITY)

    treadmill = simpy.Resource(env, capacity=NUM_TREADMILLS)
    barbell_set = simpy.Resource(env, capacity=NUM_BARBELL_SETS)
    dumbbell_set = simpy.Resource(env, capacity=NUM_BARBELL_SETS)
    kettlebell_set = simpy.Resource(env, capacity=NUM_KETTLEBELL_SETS)
    stretching_set = simpy.Resource(env, capacity=NUM_STRETCHING_SETS)
    pull_up_frame = simpy.Resource(env, capacity=NUM_PULL_UP_FRAMES)
    leg_press = simpy.Resource(env, capacity=NUM_LEG_PRESSES)
    leg_extension_machine = simpy.Resource(env, capacity=NUM_LEG_EXTENSION_MACHINES)
    stationary_bicycle = simpy.Resource(env, capacity=NUM_STATIONARY_BICYCLES)
    fitness_ball = simpy.Resource(env, capacity=NUM_FITNESS_BALLS)
    cable_machine = simpy.Resource(env, capacity=NUM_CABLE_MACHINES)
    lever_machine = simpy.Resource(env, capacity=NUM_LEVER_MACHINES)

    st_client_with_personal_t_counter = 0
    st_client_with_group_t_counter = 0
    vip_client_with_personal_t_counter = 0
    vip_client_with_group_t_counter = 0
    st_client_without_t_counter = 0
    vip_client_without_t_counter = 0

    run_results = {}

    run_customers_handled = 0

    def main_loop():
        for _ in range(MAX_CLIENTS):
            # Generate client characteristics
            take_a_trainer = False
            day_plan = "leg_day"

            vip_probability = 0.3
            membership_type = 2

            # VIP pack is value of membership_type 1 and standard pack is value of membership_type 2
            if random.random() <= vip_probability:
                membership_type = 1

            with gym_capacity.request(priority=membership_type) as req:
                yield req

                # Create processes for client's gym visit and trainers obstructions
                env.process(attend_gym(take_a_trainer, day_plan, membership_type))


            # Generate random inter-arrival times
            random_sample = random.expovariate(1.0 / MEAN_ARRIVAL_TIME)
            yield env.timeout(random_sample)

    # Start the main simulation loop
    env.process(main_loop())
    env.process(obstruct_trainer(personal_coach, OBSTRUCT_TRAINER_TIME, OBSTRUCT_TRAINER_FREQUENCY,
                                 "Personal coach"))
    env.process(obstruct_trainer(group_coach, OBSTRUCT_TRAINER_TIME, OBSTRUCT_TRAINER_FREQUENCY,
                                 "Group coach"))
    env.run(SIM_TIME)

    # print(run_results)

    mean_results = {}

    for key, value in run_results.items():
        mean_results[key] = sum(value)/len(value)

    total_runs_results[run_number] = mean_results

    total_customers_handled.append(run_customers_handled)

    income_st_clients_with_p_t = st_client_with_personal_t_counter * (BASIC_PACK_PER_DAY + PERSONAL_TRAINER_PER_DAY)
    income_st_clients_with_g_t = st_client_with_group_t_counter * (BASIC_PACK_PER_DAY + GROUP_TRAINER_PER_DAY)
    income_vip_clients_with_p_t = vip_client_with_personal_t_counter * (VIP_PACK_PER_DAY + PERSONAL_TRAINER_PER_DAY)
    income_vip_clients_with_g_t = vip_client_with_group_t_counter * (VIP_PACK_PER_DAY + GROUP_TRAINER_PER_DAY)
    income_st_clients_without_t = st_client_without_t_counter * BASIC_PACK_PER_DAY
    income_vip_clients_without_t = vip_client_without_t_counter * VIP_PACK_PER_DAY

    # Concatenate income results to the DataFrame
    df_to_add = pd.DataFrame({"Run number": [run_number],
                              "Income from standard clients with personal trainer":
                                  [round(income_st_clients_with_p_t, 2)],
                              "Income from standard clients with group trainer": [round(income_st_clients_with_g_t, 2)],
                              "Income from standard clients without trainer": [round(income_st_clients_without_t, 2)],
                              "Income from VIP clients with personal trainer": [round(income_vip_clients_with_p_t, 2)],
                              "Income from VIP clients with group trainer": [round(income_vip_clients_with_g_t, 2)],
                              "Income from VIP clients without trainer": [round(income_vip_clients_without_t, 2)]})
    df_to_add.set_index("Run number", inplace=True)
    income_df = pd.concat([income_df, df_to_add])

print()
print("RESULTS")
print("-------")

all_values = {"Personal coach": [], "Group coach": [], "Treadmill": [],
              "Bicycle": [], "Stretching set": [], "Barbell set": [],
              "Dumbbell set": [], "Kettlebell set": [], "Leg press": [],
              "Leg extension machine": [], "Cable machine": [], "Lever machine": [],
              "Pull up frame": [], "Fitness ball": []}

# Iter through inner dictionary
for inner_dict in total_runs_results.values():
    # Iterate over the keys in each inner dictionary
    for key, value in inner_dict.items():
        # Check if the key is valid and add the value to the corresponding list
        if key in all_values:
            all_values[key].append(value)

mean_values_result = {key: sum(value)/len(value) for key, value in all_values.items()}

equipment_names = []
waiting_times = []
# Display the mean waiting times for equipment and trainers
for key, value in mean_values_result.items():
    print(f"{key} total mean waiting time: {round(value, 2)}")
    equipment_names.append(key)
    waiting_times.append(value)

columns_names = income_df.columns.to_list()
mean_income_df_values = income_df.mean().to_list()

total_sim_income = sum(mean_income_df_values)

total_mean_customers_handled = sum(total_customers_handled)/len(total_customers_handled)

print()
print(f"Simulation time: {SIM_TIME//60}h.{SIM_TIME%60}m.")
print(f"Mean total income for {NUM_RUNS} simulations: {round(total_sim_income, 2)} usd")
print(f"Mean total customers handled: {round(total_mean_customers_handled)}")

# Create a bar chart for mean waiting times
plt.figure(figsize=(12, 6))
plt.bar(equipment_names, waiting_times, color='skyblue')
plt.xlabel("Equipment/Trainer")
plt.ylabel("Mean Waiting Time (minutes)")
plt.title("Mean Waiting Time for Different Equipment and Trainers")
plt.xticks(rotation=45, ha="right")

# Create a bar plot for income distribution
plt.figure(figsize=(10, 6))
plt.bar(columns_names, mean_income_df_values, color='skyblue')
plt.xticks(rotation=45, ha='right')

# Add labels and title
plt.xlabel("Income Categories")
plt.ylabel("Income Amount (USD)")
plt.title("Income Distribution by Category")

# Display the charts
plt.tight_layout()
plt.show()
