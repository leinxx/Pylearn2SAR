#!/usr/bin/env python
#All kinds of test functions to debug, prob the cnn

import cPickle
import theano
import theano.tensor as T
import Image
import numpy as np


def check_ann_cost(ann,datapath):
    with open(datapath) as f:
        datasets = cPickle.load(f)
    num_channels = ann.input_space.num_channels
    num_rows = ann.input_space.shape[0]
    num_cols = ann.input_space.shape[1]
    train_x, train_y = datasets[0]
    valid_x,valid_y = datasets[1]
    test_x, test_y = datasets[2]
    train_y = train_y.reshape(len(train_y),1).astype('float64')/10.
    valid_y = valid_y.reshape(len(valid_y),1).astype('float64')/10.
    test_y = test_y.reshape(len(test_y),1).astype('float64')/10.
    train_x = np.transpose(train_x.reshape(len(train_x),num_channels,num_rows,num_cols),(0,2,3,1))
    valid_x = np.transpose(valid_x.reshape(len(valid_x),num_channels,num_rows,num_cols),(0,2,3,1))
    test_x =  np.transpose( test_x.reshape(len(test_x), num_channels,num_rows,num_cols),(0,2,3,1))

    cost_train = cal_cost_from_np(ann,train_x,train_y)
    cost_test = cal_cost_from_np(ann,test_x,test_y)
    cost_valid = cal_cost_from_np(ann,valid_x,valid_y)
    return {'cost_train':cost_train,'cost_test':cost_test,'cost_valid':cost_valid}

def cal_cost_from_np(ann,x,Y):
    X = ann.get_input_space().make_batch_theano()
    y = ann.fprop(X)
    f = theano.function([X],y)
    bs = ann.batch_size
    pred = np.zeros((len(x),1))
    for i in range(0,len(x),bs):
        pred[i:i+bs,:] = f(x[i:i+bs,:,:,:])
    return np.mean((pred-Y)**2)

def export_weights(ann):
    '''
    input: ann, could be ann or ann file
    '''
    if isinstance(ann,str):
        with open(ann) as f:
            ann = cPickle.load(f)
    layers = ann.layers
    t=1
    for layer in layers:
        w = layer.get_weights_topo()
        shape = w.shape
        w_flat = np.zeros((shape[3]*shape[1]+shape[3]-1,shape[0]*shape[2]+shape[0]-1))
        w_flat[:] = np.nan
        for i in range(shape[0]):
            for j in range(shape[3]):
                w_flat[j+j*shape[1]:j+(j+1)*shape[1],i+i*shape[2]:i+(i+1)*shape[2]]=w[i,:,:,j]
        Image.fromarray(w_flat).save('w'+str(t)+'.tif')
        t = t+1

def export_intermedian_results(ann,fns,ofns):
    '''
    export the out put of each layer given the image file fns
    parameters:
        ann: the file name of the ann pkl file, or the ann object
        fns: the name of the image file to be predicted
    '''
    X = ann.get_input_space().make_batch_theano()
    for i in range(len(ann.layers)):

        y = ann.layers[i].fprop(X)
        f = theano.function([X],y)
        window = ann.get_input_space().shape[0]



def email_notice(msg, to_add = 'alphaleiw@gmail.com'):
    if to_add is None:
        to_add = 'alphaleiw@gmail.com'

    import smtplib

    from_add = "leindummy@gmail.com"
    username = "leindummy"
    password = "uwaterloodummy"

# Start the server using gmail's servers
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username, password)
    server.sendmail(from_add, to_add, msg)

# Log off
    server.quit()


def sms_notice(msg,to_add='+15195001886'):
    from twilio.rest import TwilioRestClient
    if to_add == 'Jenny':
        to_add = '+15197299287'
# Your Account Sid and Auth Token from twilio.com/user/account
    account_sid = "ACe42ecaccd797824f0ec3998926aea336"
    auth_token  = "b9dba936dd36227019b5cfbd37f4fc3d"
    client = TwilioRestClient(account_sid, auth_token)

    message = client.messages.create(
        body=msg,
        to=to_add,    # Replace with your recipient's phone number
        from_="+12268871484") # Replace with your Twilio number
    print message.sid # Check send confirmation


def plot_cost(fann):
    import matplotlib.pyplot as plt
    objs = []
    colors = "bgrcmykw"
    colorindex=0
    plt.figure(1)
    with open(fann) as f:
            ann = cPickle.load(f)

    train_obj = ann.monitor.channels['train_objective'].val_record
    test_obj = ann.monitor.channels['test_objective'].val_record
    valid_obj = ann.monitor.channels['valid_objective'].val_record
    objs.append([np.array(train_obj).min(),np.array(test_obj).min(),np.array(valid_obj).min()])
    plt.plot(train_obj,'-',label='train_obj:'+str(window), c=colors[colorindex])
    plt.plot(test_obj,'--',label = 'test_obj:'+str(window),c = colors[colorindex])
    plt.plot(valid_obj,'-',label = 'valid_obj'+str(window), c = colors[colorindex],marker='.')

def plot_ann_errors():
    '''
    plot the errors of anns with different window size
    '''
    import matplotlib.pyplot as plt
    objs = []
    colors = "bgrcmykw"
    colorindex=0
    plt.figure(1)
    basedir = '../output/'
    for window in range(28,53,4):
        fann = basedir+'train_with_2010_{0}/train_with_2010_{0}.pkl'.format(window)
        with open(fann) as f:
            ann = cPickle.load(f)

        train_obj = ann.monitor.channels['train_objective'].val_record
        test_obj = ann.monitor.channels['test_objective'].val_record
        valid_obj = ann.monitor.channels['valid_objective'].val_record
        objs.append([np.array(train_obj).min(),np.array(test_obj).min(),np.array(valid_obj).min()])
        plt.plot(train_obj,'-',label='train_obj:'+str(window), c=colors[colorindex])
        plt.plot(test_obj,'--',label = 'test_obj:'+str(window),c = colors[colorindex])
        plt.plot(valid_obj,'-',label = 'valid_obj'+str(window), c = colors[colorindex],marker='.')
        colorindex += 1

    plt.legend()
    objs = np.array(objs)
    plt.figure(2)
    plt.plot(objs[:,0],'r',label='best train obj')
    plt.plot(objs[:,1],'g',label='best test obj')
    plt.plot(objs[:,2],'b',label='best valid obj')
    plt.legend()
    plt.show()
    return objs

if __name__=='__main__':
    plot_ann_errors()
