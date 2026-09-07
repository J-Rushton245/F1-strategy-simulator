import logging
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fastf1
import fastf1.plotting as plotting
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import random
import math

#####SESSION VARIABLES#####
year = 2026
track = 3 #round number or track identifier - 3 is Japan which has behaved best with my simulation (to help focus on strategy simulation developments)
session_type = 'R' #'R' for race, practice use 'FP2', if sprint weekend use 'FP1'
session = fastf1.get_session(year, track, session_type)
session.load()
total_laps = session.total_laps
rand_fact = False #adds factor of randomisation to laptimes (to remove perfect consistency)
sim_option = 3 # 1:Fastest Clean Air Strategy 2:Race Sim, all drivers predetermined strat 3:Monte Carlo sims

#####HARDCODED VALUES#####
pit_loss = 24 #assumed pit stop time (temporary for Sazuka) - get standard pit losses for each track
overtaking_delta = 0.25 #rough min pace advantage (s) to overtake
start_fuel = 110 #assumed initial fuel load (kg)
fuel_burn = 1.7 #kg/lap (estimated) - for race sim fuel estimation
fuel_loss = 0.03 #fuel loss (s/kg)
max_dirty_air = 0.35 #maximum dirty air loss (s)
dirty_air_effect = 2 #time behind car ahead in which dirty air impacts laptime

logging.disable(logging.INFO) #disables logging data while keeping warnings
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme="fastf1") #allows matplotlib to work with fastf1 colour 

#definition of classes
class Driver:
    def __init__(self, name, team, deg_fact, base_pace, race_time, laptime, strat, pit_laps, pit_windows, position, tyre_age):
        self.name = name
        self.team = team
        self.deg_fact = deg_fact
        self.base_pace = base_pace
        self.race_time = 0.0
        self.laptime = laptime
        self.strat = None
        self.pit_laps = None
        self.pit_windows = None
        self.position = position
        self.tyre_age = tyre_age #deg of tyres (in terms of laps - not strictly 1 lap == 1 tyre age!), for scaling with dirty air:

    def get_driver_lap(self, tyre, fuel_time): #getting laptimes during race sims
        deg = tyre.get_deg(self.tyre_age)
        lap = self.base_pace[tyre.compound] + fuel_time + self.deg_fact * deg
        self.tyre_age += 1 #updating tyre age simultaneously
        return lap

    def get_dirty_air(self, gap): #calculates dirty air loss to car ahead
        if 0 <= gap <= 2: #assuming dirty air impacts up to 2s behind
            loss = max_dirty_air * (1 - gap/2)
        else:
            loss = 0

        #adding small tyre wear increase in dirty air
        if gap < dirty_air_effect/4:
            dirty_wear = 0.02
        elif gap < dirty_air_effect/2:
            dirty_wear = 0.01
        elif gap < dirty_air_effect:
            dirty_wear = 0.005
        else:
            dirty_wear = 0
        self.tyre_age += dirty_wear
        return loss

class Tyre:
    def __init__(self, compound, a, b):
        self.compound = compound
        self.a = a
        self.b = b
    
    def get_deg(self,tyre_age):
        return self.a*tyre_age**2 + self.b*tyre_age

class Stint:
    def __init__(self, driver, tyre, start_lap, end_lap, time):
        self.driver = driver
        self.tyre = tyre
        self.start_lap = start_lap 
        self.end_lap = end_lap
        self.time = time

    def get_laptime(self, fuel_time, deg): #from stint calculations
        laptime = self.driver.base_pace[self.tyre.compound] + fuel_time + self.driver.deg_fact * deg
        return laptime

    def get_stint_time(self):
        if self.end_lap != total_laps:
            self.time += pit_loss #pit loss counts at end of stint
        for i in range(self.end_lap-self.start_lap+1):
            fuel_remaining = start_fuel*(1-((self.start_lap+i)/total_laps))
            fuel_time = fuel_remaining*fuel_loss #loss to fuel remaining
            deg = self.tyre.get_deg(i) #loss to tyre deg ----------------------------- Add some sort of fuel interaction here (make this wear and compound specific)
            laptime = self.get_laptime(fuel_time, deg)
            if rand_fact == True:
                laptime += random.randint(-200,200) / 1000
            #print(self.start_lap+i,laptime)
            self.time += laptime
        return self.time

def race_output(drivers):
    winner = min(driver.race_time for driver in drivers.values())
    position = 0
    print(f"Simulated {session.event.OfficialEventName}")
    print("Position Name Race_time Strategy Optimal_pit Pit_window")
    for driver in sorted(drivers.values(), key=lambda d: d.race_time): #formatting race times
        position += 1
        if position == 1:
            td = pd.Timedelta(seconds=driver.race_time)
            hours = td.components.hours
            minutes = td.components.minutes
            seconds = td.components.seconds
            milliseconds = td.components.milliseconds
            print(f"{position}.{driver.name} {hours}:{minutes:02}:{seconds:02}.{milliseconds:03} {driver.strat} {driver.pit_laps} {driver.pit_windows}")
        else:
            print(f"{position}.{driver.name} +{driver.race_time-winner:.2f}s {driver.strat} {driver.pit_laps} {driver.pit_windows}")

def one_stop_sim(driver, tyres, strat): #finding optimal pit lap for 1 stop + pit window
    race_times = []
    optimal_pit = [0,None] #fastest pit lap for given strategy (pit lap is classed as the lap the pitlane is entered - not lap exited)
    start = math.floor(total_laps/4) #earliest pit lap being tested (to improve iteration efficiency)
    end = math.ceil(3*total_laps/4) + 1 #latest pit lap being tested
    for i in range(start,end):
        stint1 = Stint(driver, tyres[strat[0].capitalize()], 1, i, 0.0)
        stint2 = Stint(driver, tyres[strat[1].capitalize()], i+1, total_laps, 0.0)
        time = stint1.get_stint_time() + stint2.get_stint_time()
        race_times.append([time,i])
    fastest_time, optimal_pit[0] = min(race_times, key=lambda x: x[0]) #finds optimal strategy
    pit_window = [row[1] for row in sorted(race_times, key=lambda x: x[0])[:7]] #getting 7 fastest pit laps (to generate pit window)
    pit_window = f"{min(pit_window)}-{max(pit_window)}"
    return fastest_time, optimal_pit, pit_window

def two_stop_sim(driver, tyres, strat): #finding optimal pit laps for each 2 stop
    fastest_time = 10000000000 #large placeholder value much larger than actual race time
    pit_laps = [0,0] #fastest pit lap for given strategy (pit lap is classed as the lap the pitlane is entered - not lap exited)
    for x in range(1,total_laps-1):
        for y in range(x,total_laps):
            stint1 = Stint(driver, tyres[strat[0].capitalize()], 1, x, 0.0)
            stint2 = Stint(driver, tyres[strat[1].capitalize()], x+1, y, 0.0)
            stint3 = Stint(driver, tyres[strat[2].capitalize()], y+1, total_laps, 0.0)
            time = stint1.get_stint_time() + stint2.get_stint_time() + stint3.get_stint_time()
            if time < fastest_time:
                fastest_time = time
                pit_laps = [x,y]
    return fastest_time, pit_laps, None

def get_strats(compounds): #Getting legal strategies with assumed tyre availability
    strats = []
    for i in range(2): 
        arr = ["SOFT", "MEDIUM", "HARD"] #assuming 2 fresh sets of each compound
        if i != 0:
            arr = ["SOFT", "MEDIUM", "MEDIUM", "HARD"] #assuming fresh sets of compounds on left available
        #print(arr)
        for x in itertools.combinations(arr, i+2):
            strat = sorted(x, key=lambda c: compounds.index(c)) #ordering starting on softer compound (generally preferred strategy for track position at race start)
            if strat not in strats: #filtering out repeating strategies (as currently no fuel affect on tyres means soft-medium == medium-soft for example)
                strats.append(strat)
    return strats

def race_sim(baselines, team_deg, global_a, global_b, compounds):
    drivers = {}
    tyres = {}
    for _, row in baselines.iterrows():
        driver = row["Driver"]
        team = row["Team"]
        deg_fact = team_deg.loc[team]
        base_laps = {}
        for compound in compounds:
            temp = compound.capitalize()
            base_laps[temp] = row[temp]
        if sim_option != 3:
            drivers[driver] = Driver(driver, team, deg_fact, base_laps, 0.0, None, None, None, None, None, 0)
        if sim_option == 3:
            grid_pos = int(session.results.loc[session.results["Abbreviation"] == driver, "GridPosition"].iloc[0])
            drivers[driver] = Driver(driver, team, deg_fact, base_laps, 0.0, 0.0, None, None, None, grid_pos, 0.0)
    for i in range(len(compounds)):
        tyre = compounds[i].capitalize()
        tyres[tyre] = Tyre(tyre, global_a[i], global_b[i])

    if sim_option == 1: #getting fastest clean air strategy (for testing strategy model)
        strats = get_strats(compounds)
        for driver in drivers.values():
            print(driver.name)
            fastest_time = 10000000000 #race time of strategy
            tyre_strat = [None, None, None] #tyre strategy
            pit_strat = [0,0] #pit strategy
            for strat in strats:
                if len(strat) == 2:
                    race_time, pit_laps, pit_window = one_stop_sim(driver, tyres, strat)
                    print(strat, pit_laps, race_time)
                elif len(strat) == 3:
                    race_time, pit_laps, pit_window = two_stop_sim(driver, tyres, strat)
                    print(strat, pit_laps, race_time)

                if race_time < fastest_time: #checking fastest strategy for driver
                    fastest_time = race_time
                    tyre_strat = strat
                    pit_strat = pit_laps
                    window = pit_window
            driver.race_time = fastest_time
            driver.strat = tyre_strat.copy()
            driver.pit_laps = pit_strat.copy()
            driver.pit_windows = window
        race_output(drivers)

    if sim_option == 2: #simulates clean air race (with all drivers on set strategy - for race pace data tests)
        for driver in drivers.values():
            driver.strat = ["Medium", "Hard"]
            driver.pit_laps = [29,None]
            pit_lap1 = driver.pit_laps[0]
            pit_lap2 = driver.pit_laps[1]
            stint1 = Stint(driver, tyres[driver.strat[0]], 1, pit_lap1, 0.0)
            stint2 = Stint(driver, tyres[driver.strat[1]], pit_lap1 + 1, total_laps, 0.0)
            driver.race_time = stint1.get_stint_time() + stint2.get_stint_time()
            print()
            #print(driver.name + " " + str(driver.race_time))
        race_output(drivers)

    if sim_option == 3: #race simulations (monte carlo), including traffic, dirty air and track events
        #iterate monte carlo sims here

        grid = sorted(drivers.values(),key=lambda x: x.position)
        strats = get_strats(compounds)
        for driver in grid:
            num = random.randint(0,len(strats)-1)
            driver.strat = strats[num]
            #simulations to find optimal pit lap
            if len(driver.strat) == 2:
                _, driver.pit_laps, driver.pit_windows = one_stop_sim(driver, tyres, driver.strat)
            elif len(driver.strat) == 3:
                _, driver.pit_laps, driver.pit_windows = two_stop_sim(driver, tyres, driver.strat)
        for i in range(1,total_laps+1):
            if i == 1:
                order = grid
            if i > 1:
                order = sorted(drivers.values(),key=lambda x: x.race_time)
            print(f"Lap {i}: {' - '.join(driver.name for driver in order)}")
            fuel_remaining = start_fuel*(1-(i/total_laps)) #fuel model here as fuel model same for each car
            fuel_time = fuel_remaining*fuel_loss #loss to fuel remaining
            for position, driver in enumerate(order, start=1):
                driver.position = position
                pit_laps = driver.pit_laps
                strat = driver.strat
                pit = False
                laptime = 0
                if i <= pit_laps[0]: #on first stint
                    tyre = tyres[strat[0].capitalize()]
                    laptime += driver.get_driver_lap(tyre, fuel_time)
                elif (i > pit_laps[0]) and (pit_laps[1] is None or i <= pit_laps[1]): #on second stint
                    tyre = tyres[strat[1].capitalize()]
                    laptime += driver.get_driver_lap(tyre, fuel_time)
                elif i > pit_laps[1]: #on third stint (if applies)
                    tyre = tyres[strat[2].capitalize()]
                    laptime += driver.get_driver_lap(tyre, fuel_time)

                if position != 1: #no dirty air for leader (currently - can include dirty air in lapped cars later down the line)
                    gap = (driver.race_time + laptime) - order[position-2].race_time
                    #print(gap, driver.get_dirty_air(gap))
                    laptime += driver.get_dirty_air(gap)
                driver.laptime = laptime
            new_order = order.copy()
            overtaken = set()
            
            for j in range(0, len(order)): #overtaking model
                overtake = False
                driver = order[j]
                if j != 0:
                    driver_ahead = order[j-1]
                    temp = driver_ahead.race_time
                    gap = driver.race_time + driver.laptime - driver_ahead.race_time
                    delta = driver.laptime - driver_ahead.laptime
                    if driver not in overtaken and driver_ahead not in overtaken:
                        delta = driver.laptime - driver_ahead.laptime
                        if delta >= -overtaking_delta:
                            pass
                        elif delta < -overtaking_delta and gap + delta < 0.2:
                            strength = -delta - overtaking_delta
                            overtake_prob = 0.1 + strength * 0.8
                            overtake_prob = np.clip(overtake_prob, 0, 0.95)
                            if random.randint(0,100)/100 <= overtake_prob:
                                overtake = True
                                print(f"{driver.name} overtakes {driver_ahead.name}")
                                overtake_time_change = random.randint(-250,250)/1000 #gain / loss due to overtaking aid / battle
                                driver.laptime += overtake_time_change
                                overtaken.add(driver)
                                overtaken.add(driver_ahead)
                                new_order[j], new_order[j-1] = new_order[j-1], new_order[j] #changing order
                                driver.race_time += driver.laptime
                                if driver_ahead.race_time <= driver.race_time: #making sure race times match order
                                    driver_ahead.race_time = driver.race_time + 0.001 #create small gap (1 hundredth)
                                driver_ahead.laptime = driver_ahead.race_time - temp #recalculating laptime of driver being overtaken

                    if overtake == False:
                        driver.race_time += driver.laptime

                    if driver_ahead.race_time > driver.race_time and overtake == False: #ensuring all switches come through overtake model
                        driver.race_time -= driver.laptime #removing laptime to match car infront (if stuck, instead of passing outside overtake model)
                        driver.race_time += driver_ahead.laptime 
                        driver.laptime = driver_ahead.laptime
                        #driver_ahead.race_time = driver.race_time + 0.001 #create small gap (1 hundredth)
                else:
                    driver.race_time += driver.laptime
            order = new_order
            for driver in order:
                if i == driver.pit_laps[0] or i == driver.pit_laps[1]: #on pit lap
                    print(f"{driver.name} pits")
                    driver.laptime += pit_loss
                    driver.race_time += pit_loss
                    driver.tyre_age = 0


        race_output(drivers)

def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)

def trimmed_mean(x): #drops top/bottom 25% 
    x = np.sort(x)
    cut = int(len(x) * 0.25)
    return np.mean(x[cut:-cut] if len(x) > 4 else x)

def assume_deltas(deltas): #fills in any delta gaps
    soft_bias = 0.45 #multiplier for gap between tyres (<0.5: mediums closer to softs, >0.5: mediums closer to hards)
    #Case 1 - Multiple deltas unknown (only if everyone ran same two compounds)
    if deltas[0] == -1 and deltas[1] == -1:
        deltas[0] = (deltas[2] / (1-soft_bias))*soft_bias
    elif deltas[0] == -1 and deltas[2] == -1:
        deltas[0] = deltas[1] * soft_bias
    elif deltas[1] == -1 and deltas[2] == -1:
        deltas[2] = (deltas[0] / soft_bias)*(1-soft_bias)

    #Case 2 - One delta unknown
    if deltas[0] == -1:
        deltas[0] = deltas[1] - deltas[2]
    elif deltas[1] == -1:
        deltas[1] = deltas[0] + deltas[2]
    elif deltas[2] == -1:
        deltas[2] = deltas[1] - deltas[0]

    return deltas

def get_tyre_deltas(times, deltas): #finds all gaps in tyre performance
    if times[0] != -1 and times[1] != -1:
        if (times[1]-times[0] > 0.1):
            deltas[0].append(times[1]-times[0])
    if times[0] != -1 and times[2] != -1:
        if (times[2]-times[0] > 0.1):
            deltas[1].append(times[2]-times[0])
    if times[1] != -1 and times[2] != -1:
        if (times[2]-times[1] > 0.1):
            deltas[2].append(times[2]-times[1]) 
    #print(deltas)
    return deltas

def driver_baselines(stint, times): #finds basline laps for each driver for each compound
    stint_clean = stint["LapTimeCorrected"]
    q1 = np.percentile(stint_clean, 25)
    q3 = np.percentile(stint_clean, 75)
    iqr = q3 - q1
    stint_clean = stint_clean[(stint_clean >= q1 - 1.5 * iqr) & (stint_clean <= q3 + 1.5 * iqr)]
    base_time = np.percentile(stint_clean, 40) #gets rough fast pace of tyre
    if stint["Compound"].iloc[0] == "SOFT": #finds the tyre of the stint
        if times[0] == -1: #replaces basetime with estimate lap
            times[0] = base_time
        else:
            times[0] = np.mean([times[0],base_time]) #incase of multiple stints on same compound
    elif stint["Compound"].iloc[0] == "MEDIUM":
        #print(times)
        if times[1] == -1:
            times[1] = base_time
        else:
            times[1] = np.mean([times[1],base_time])
    elif stint["Compound"].iloc[0] == "HARD":
        if times[2] == -1:
            times[2] = base_time
        else:
            times[2] = np.mean([times[2], base_time])
    return times

def baseline_estimate(times, deltas):
    if -1 in deltas:
        deltas = assume_deltas(deltas)
    d_sm, d_sh, d_mh = deltas
    #case 1: No compound time
    if times == [-1,-1,-1]:
        return times
    #case 2: One compound time
    if times[0] == -1 and times[1] == -1: #don't know soft or medium baselines
        times[0] = times[2] - d_sh
        times[1] = times[2] - d_mh
    elif times[0] == -1 and times[2] == -1:
        times[0] = times[1] - d_sm
        times[2] = times[1] + d_mh
    elif times[1] == -1 and times[2] == -1:
        times[1] = times[0] + d_sm
        times[2] = times[0] + d_sh
    #case 3: Two compound times
    if times[0] == -1:
        times[0] = times[1] - d_sm
    elif times[1] == -1:
        times[1] = times[0] + d_sm
    elif times[2] == -1:
        times[2] = times[1] + d_mh
    #validation checks 
    if (times[1] > times[2]) and (abs(times[1] + d_mh - times[2]) > 0.0001): #predicting if time is distorted (i.e. if hard baseline is quicker than medium baseline)
        times[1] = times[2] - d_mh
    if (times[0] > times[1]) and (abs(times[0] + d_sm - times[1]) > 0.0001):
        times[0] = times[1] - d_sm

    return times

def tyre_fit(stint): #quadratic tyre deg model
    if len(stint) < 3:
        return None
    x = stint["TyreLife"].to_numpy()
    y = stint["LapTimeCorrected"].to_numpy()
    a, b, c = np.polyfit(x, y, 2)
    lap_fit = a*x**2 + b*x + c #creates quadratic tyre graph
    r2 = r2_score(y, lap_fit)
    if session_type != 'R':
        return (a, b, c, r2)
    #print(stint["Compound"].iloc[0])
    if stint["Compound"].iloc[0] == "SOFT" and -b/(2*a)>5: #checking any soft runs with too much tyre improvement early
        #print("Done")
        return None
    if r2 > 0.5 and a > 0 and len(stint) >= 8: #filters out inaccurate / short stints (a>0 prevents tyre improvement, -b/(2*a)<5 can be used as well preventing too much tyre improvement early stint - especially for soft tyres)
        return (a, b, c, r2)
    else:
        return None

#fuel correction of laptimes will help to isolate effect of other factors on laptime (mainly tyre wear)
def fuel_correction(laptime, lap, fuel_load): #fuel correction for laptimes
    if session_type == 'R':
        fuel_remaining = start_fuel*(1-(lap/total_laps)) #assuming fuel burns linearly to 0
    else:
        fuel_remaining = fuel_load - fuel_burn * (lap - 1)
    correction = fuel_remaining*fuel_loss
    return (laptime - correction) #fuel corrected laptime

def race_data(): #gathers data to run analysis
    drivers = session.results["Abbreviation"].tolist()
    compounds = ["SOFT","MEDIUM","HARD"]
    
    #getting stint data for tyre deg
    status_codes = ["2","4","5","6","7"] #status codes for 'bad' track conditions (yellow flag, SC, Red flag, VSC deployed, VSC end)
    deg_data = [] #tyre data
    baselines = [] #base driver laptimes
    deltas = [[],[],[]] #tyre deltas [soft-med,soft-hard,med-hard]
    for driver in drivers: #iterating through drivers
        driver_times = [-1,-1,-1] #base laptimes for driver [soft, medium, hard]
        laps = session.laps.pick_drivers(driver).pick_accurate().reset_index() #picks accurate laptimes of drivers
        if session_type != 'R': #ie if practice session
            valid_stints = [] #race sim stints in practice sessions
            for stint in laps["Stint"].unique():
                stint_laps = laps[laps["Stint"] == stint].copy()
                print(stint_laps)
                if (len(stint_laps) >= 6 and stint_laps["TyreLife"].max() >= 8 and stint_laps["PitOutTime"].isna().all() and stint_laps["PitInTime"].isna().all()):
                   valid_stints.append(stint)
            print(valid_stints)
            laps = laps[laps["Stint"].isin(valid_stints)].reset_index()
        laps = laps[(laps["Deleted"] == False)] #removes deleted laps
        for code in status_codes:
            laps = laps[~laps["TrackStatus"].str.contains(code, na=False)] #removes laps under 'bad' track conditions
        for stint in laps["Stint"].unique():
            stint_laps = laps[laps["Stint"] == stint].copy()
            median = stint_laps["LapTime"].median()
            stint_laps = stint_laps[stint_laps["LapTime"] < median*1.03] #filtering out slower laps (within +3% of median)  
            stint_laps["LapTimeSeconds"] = stint_laps["LapTime"].dt.total_seconds()
            if session_type == 'R':
                stint_laps["LapTimeCorrected"] = [fuel_correction(lap_time, lap_number, None) for lap_time, lap_number in zip(stint_laps["LapTimeSeconds"], stint_laps["LapNumber"])]
            else:
                fuel_load = fuel_burn*stint_laps["LapNumber"].max() + 10
                stint_laps["LapTimeCorrected"] = [fuel_correction(lap_time, lap_number, fuel_load) for lap_time, lap_number in zip(stint_laps["LapTimeSeconds"], stint_laps["LapNumber"])]
            data = tyre_fit(stint_laps) #fits tyre model
            if data is not None:
                a, b, c, r2 = data
                deg_data.append({"Driver": driver, "Team": session.get_driver(driver)["TeamName"], "Compound": stint_laps["Compound"].iloc[0], "a": a, "b": b, "c": c, "r2": r2, "Laps": len(stint_laps)}) #stores tyre data
            #print(stint_laps)
            driver_times = driver_baselines(stint_laps, driver_times) #adds estimated baselines from stint to list
            #print(deg_data)
        baselines.append({"Driver": driver, "Team": session.get_driver(driver)["TeamName"], "Times": driver_times})
        deltas = get_tyre_deltas(driver_times, deltas) #finding difference in compounds

    new_deltas = []
    for delta in deltas: #finding mean difference in compounds
        if len(delta) != 0:
            new_deltas.append(trimmed_mean(delta))
        else:
            new_deltas.append(-1)

    #extrapolating gaps in driver and team baselines and limiting teammate spreads due to bad stints
    temp_baselines = []
    baselines = pd.DataFrame(baselines) #tyre baselines for each driver
    for _, row in baselines.iterrows():
        driver = row["Driver"]
        driver_times = baseline_estimate(row["Times"], new_deltas) #estimates any gaps in baselines
        temp_baselines.append({"Driver": driver, "Team": session.get_driver(driver)["TeamName"], "Soft": driver_times[0], "Medium": driver_times[1], "Hard": driver_times[2]})
    baselines = temp_baselines
    deg_data = pd.DataFrame(deg_data) #tyre deg info
    baselines = pd.DataFrame(baselines) #tyre baselines for each driver
    #print(baselines)
    team_baselines = baselines.groupby("Team")[["Soft","Medium","Hard"]].agg(lambda x: x[x > 0].min()) #fastest team time for each tyre (ignoring any -1s)
    #print(team_baselines)
    for idx, row in baselines.iterrows():
        #print(row["Driver"])
        team = row["Team"]
        for compound in compounds:
            temp = compound.capitalize()
            team_val = team_baselines.loc[team, temp]
            gap = row[temp] - team_val
            if gap <= 0.4: #within roughly 0.4s
                k = 1.0
            elif gap <= 0.8:
                k = 0.5
            else:
                k = 0.4
            if team_val < (gap*-1): #if the driver has no representative lap time
                k = 0 #set their value to the team mean
            #print(k)
            #print(baselines[temp])
            #print(team_val + k*gap)
            baselines.loc[idx, temp] = team_val + k*gap #shrunk laptimes closer to team mean
    #print(baselines)
    #getting team deg multipliers with global model coefficients
    team_deg = deg_data.groupby(["Team", "Compound"]).apply(lambda x: pd.Series({"a": weighted_mean(x["a"], x["r2"]*x["Laps"]), "b": weighted_mean(x["b"], x["r2"]*x["Laps"]), "Laps": weighted_mean(x["Laps"], x["r2"]*x["Laps"])})) #weighted mean for each team based on accuracy of model and stint length
    global_deg = deg_data.groupby(["Compound"])[["a", "b", "r2", "Laps"]].apply(lambda x: pd.Series({"a": weighted_mean(x["a"], x["r2"]*x["Laps"]), "b": weighted_mean(x["b"], x["r2"]*x["Laps"])})) #gets global a and b for each compound
    global_a = compounds.copy() #global a values for each compound (soft,medium,hard)
    global_b = compounds.copy() #global b values for each compound (soft,medium,hard)
    print(team_deg)
    #print(global_deg)
    for i in range(len(compounds)):
        try:
            global_a[i] = global_deg.loc[global_a[i], "a"]
            global_b[i] = global_deg.loc[global_b[i], "b"]
        except:
            global_a[i] = 0
            global_b[i] = 0
    #finds missing models using soft = 2*medium - hard, for a and b (temporary fix)
    if global_a.count(0) > 1 and global_b.count(0) > 1:
        print("Not enough data for dry prediction")
        return
    if global_a[0] == 0 and global_b[0] == 0:
        global_a[0] = 2*global_a[1] - global_a[2]
        global_b[0] = 2*global_b[1] - global_b[2]
    elif global_a[1] == 0 and global_b[1] == 0:
        global_a[1] = (global_a[0] + global_a[2])/2
        global_b[1] = (global_b[0] + global_b[2])/2
    elif global_a[2] == 0 and global_b[2] == 0:
        global_a[2] = 2*global_a[1] - global_a[0]
        global_b[2] = 2*global_b[1] - global_b[0]
    #print(global_a)
    #print(global_b)
    stint_length = team_deg["Laps"].mean()
    avg_stint_deg = [] #global model deg for each compound at average stint length (soft,medium,hard)
    for a,b in zip(global_a, global_b):
        avg_stint_deg.append(a*stint_length*2 + b) #using derivative
        #avg_stint_deg.append(a*stint_length**2 + b*stint_length)
    #print(stint_length)
    #print(avg_stint_deg)
    for x, row in team_deg.iterrows(): #obtaining team deg multipliers (at average stint length)
        compound = x[1]
        if compound == compounds[0]:
            team_deg.loc[x, "a"] = (row["a"]*stint_length*2 + row["b"])/ avg_stint_deg[0] #using derivative
        elif compound == compounds[1]:
            team_deg.loc[x, "a"] = (row["a"]*stint_length*2 + row["b"])/ avg_stint_deg[1]
        elif compound == compounds[2]:
            team_deg.loc[x, "a"] = (row["a"]*stint_length*2 + row["b"])/ avg_stint_deg[2]
        #if compound == compounds[0]:
            #team_deg.loc[x, "a"] = (row["a"]*stint_length**2 + row["b"]*stint_length)/ avg_stint_deg[0]
        #elif compound == compounds[1]:
            #team_deg.loc[x, "a"] = (row["a"]*stint_length**2 + row["b"]*stint_length)/ avg_stint_deg[1]
        #elif compound == compounds[2]:
            #team_deg.loc[x, "a"] = (row["a"]*stint_length**2 + row["b"]*stint_length)/ avg_stint_deg[2]
    team_deg = team_deg.rename(columns={"a":"team_multiplier"})
    team_deg = team_deg.groupby(level="Team")["team_multiplier"].mean()
    teams = session.results["TeamName"].unique() #obtaining list of teams in session
    team_deg = team_deg.reindex(teams, fill_value=1.0) #filling any missing teams (i.e. ones that don't have good enough quality tyre deg data) (assumes this deg is average)
    team_deg = 1 + 0.5*(team_deg-1) #shrinkage factor to compress tyre wear spread
    print(team_deg)
    team_deg = team_deg.clip(0.75,1.25) #cleaning field spread

    print(baselines)
    print(team_deg)
    print(global_a)
    print(global_b)
    race_sim(baselines, team_deg, global_a, global_b, compounds)
    
                
race_data()

    