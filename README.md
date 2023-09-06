# Gym-simulation
![block_schema](https://github.com/WizzVard/fitness-center-simulation/assets/116277163/029b5a6b-fe35-44da-8cf9-f22d4471b094)

This descrete-event simulation created with SimPy python library. The goal is to increase overall gym income by increasing amount of handled customers.
The goal is to increase the total income of the gym by increasing the number of clients served. When the customer arrives, he chooses one of two packages: 
Standard and VIP. He then decides whether to hire a coach or not, and if so, there are personal and group coaches. If the customer does not take the coach, 
there are 3 workout plans: leg day, body day and cardio day. Each of these days has different equipments.
We calculate the waiting time and income for each coach and equipment depending on the amount of equipment and package cost, then plot the results on the graphs. 

![waiting_time](https://github.com/WizzVard/fitness-center-simulation/assets/116277163/ca6ed6d5-9efc-41c0-a251-0268d25c14e2)
![income_distribution](https://github.com/WizzVard/fitness-center-simulation/assets/116277163/1869e968-235f-4a54-a94c-2265fc96a825)

Presented results are optimal for the number of instructors, equipment and income. 
All information on quantity and cost of material was taken from different sources and can be not accurate.
