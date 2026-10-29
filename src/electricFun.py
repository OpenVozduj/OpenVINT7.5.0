import math

def electricPow(rpm, torque, motor):
    voltage = motor['voltage']               
    kv = motor['kv']
    resistance = motor['resistance'] 
    i00 = motor['i00']             
    beta = motor['beta']              
    gamma = motor['gamma']  
    eta0 = motor['eta0']               
    ic = motor['ic']  
    d_max = motor['d_max']
    
    # 1. Constante de par inicial y velocidad angular
    kt0 = 60.0 / (2.0 * math.pi * kv)
    omega = rpm * math.pi / 30.0

    # 2. Par de pérdidas internas y par electromagnético total
    i0_eq = i00 + beta * omega
    loss_torque = kt0 * i0_eq
    electromagnetic_torque = torque + loss_torque

    # 3. Corriente equivalente del motor
    denominator = kt0 - gamma * electromagnetic_torque
    if denominator <= 0:
        # raise ValueError(
        #     "Punto de funcionamiento no alcanzable: "
        #     "Kt0 - gamma * Q <= 0."
        # )
        battery_power = 100000
    else:

        motor_current = electromagnetic_torque / denominator
        if motor_current == 0:
            # raise ValueError(
            #     "Corriente nula: se requiere un modelo separado "
            #     "para el funcionamiento en vacío."
            # )
            battery_power = 100000
    
        else:
            # 4. Kt efectivo, fuerza contraelectromotriz y tensión del motor
            kt = kt0 / (1.0 + gamma * motor_current)
            back_emf = kt * omega
            motor_voltage = back_emf + motor_current * resistance
            voltage_ratio = motor_voltage / voltage
        
            if voltage_ratio > d_max:
                # raise ValueError(
                #     f"Tensión insuficiente: D={voltage_ratio:.4f}, "
                #     f"máximo permitido={d_max:.4f}."
                # )
                battery_power = 100000
            
            else:
        
                # 5. Balance de potencia del motor
                shaft_power = torque * omega
                rotation_loss = loss_torque * omega
                copper_loss = motor_current**2 * resistance
                motor_input_power = shaft_power + rotation_loss + copper_loss
            
                # 6. Regulador ESC y batería
                esc_efficiency = -eta0 * math.expm1(-motor_current / ic)
                if esc_efficiency <= 0:
                    # raise ValueError(
                    #     "No se pudo calcular una eficiencia positiva del ESC."
                    # )
                    battery_power = 100000
                
                else:
                    battery_power = motor_input_power / esc_efficiency
                    battery_current = battery_power / voltage
                    esc_loss = battery_power - motor_input_power

    return {
        "battery_power_W": battery_power
           }    

def battery_mass(W_e_u, W_e_l, t, gamma):
    gamma_min = gamma * 60
    m_bat = (W_e_u + W_e_l)*t/gamma_min
    return m_bat
    
    
    
    