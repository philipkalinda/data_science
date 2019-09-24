import boto3
role_arn = 'arn:aws:iam::047625233815:role/dp-np-boiler-iq-source'
role_name = 'biq-data-science'

boto3.client('sts').assume_role(RoleArn=role_arn, RoleSessionName=role_name)['Credentials']
