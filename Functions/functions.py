
def convert_sqft_to_num(x):
    try:
        float_x = float(x)  # case where x is of type '234.3' which means we can directly covert to float
        return float_x

    except:

        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(
                tokens[1])) / 2  # tokens are in string format so we must convert to float to calculate mean
        # else(cases where we dont have a range min-max the length of tokens = 1 since those values do not contain '-')
        tokens = x.split('Sq')
        if len(tokens) == 2:  # cases '132Sq. Yards' or '716Sq. Meter'
            if 'Yards' in tokens[1]:  # case '132Sq. Yards'
                yards = float(tokens[0])
                sqft = yards / 0.11111111
                return sqft
            elif 'Meter' in tokens[1]:  # case '716Sq. Meter'
                meters = float(tokens[0])
                sqft = meters / 0.09290304
                return sqft

        else:  # cases '6Acres', '24Guntha', '3Cents', '1Grounds'
            if 'Acres' in x:
                tokens = x.split('Acres')
                acres = float(tokens[0])
                sqft = acres / 0.00002296
                return sqft

            elif 'Guntha' in x:
                tokens = x.split('Guntha')
                guntha = float(tokens[0])
                sqft = guntha / 0.00000003587
                return sqft

            elif 'Cents' in x:
                tokens = x.split('Cents')
                cents = float(tokens[0])
                sqft = cents / 0.0023
                return sqft

            elif 'Grounds' in x:
                tokens = x.split('Grounds')
                grounds = float(tokens[0])
                sqft = grounds / 0.00041666666666667
                return sqft

        return None
