## 1. Setup the lab environment. Connect developer host and target machines in the same local network
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/hosttargetsetup.png)

In host, follow the steps to setup ip address in windows’s Control Panel\All Control Panel Items\Network Connections
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/setuphostip.png)
In Linux target, use the following commands to configure Ethernet IP of target machine.
	ifconfig {Ethernet interface name} 192.168.9.22 netmask 255.255.255.0
	ping 192.168.9.11
You should receive the ping response if you successfully setup the network connection between host and target.

## 2. Setup the password-less SSH connection to access the target
Download required applications on windows host’s folder (for e.g. c:\puttyutilities)

https://the.earth.li/~sgtatham/putty/latest/w64/pscp.exe
https://the.earth.li/~sgtatham/putty/latest/w64/plink.exe
https://the.earth.li/~sgtatham/putty/latest/w64/pageant.exe
https://the.earth.li/~sgtatham/putty/latest/w64/puttygen.exe

Make sure you include c:\puttyutilities in %PATH% environment variable.
![image of the output]( https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/setuppath.png)
Generate RSA key pairs on Linux target device by the following commands with root id.
	ssh-keygen –t rsa
	cat .ssh/id_rsa.pub >> .ssh/authorized_keys

Download /root/.ssh/id_rsa to windows host and convert it to .ppk format by using puttygen.exe
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/convertprivatekey.png)

Attach the key to ssh agent by using pageant.exe
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/addkeypageant.png)

Use the following command to test if password-less ssh works or not
    plink root@192.168.9.22 ls

## 3. Build the kernel drivers on target machine.
1)	Install the entire VTune on your target - install_GUI.sh and build the drivers during the installation(prework)
2)	Install the linux target package on your linux target. Untar the downloaded target package

## 4. Setup the Intel® VTune Amplifier 2019 software package for windows host
Download the Intel® VTune Amplifier 2019 software package via https://registrationcenter.intel.com/en/products/
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/downloadvtunewinhost.png)
Install the downloaded package(VTune_Amplifier_xxxx_*.exe) on your windows host. You will be asked for a license key during the installation. 

## 5. Start the VTune to perform the remote performance profiling.
Launch VTune with administrator privilege 
New a project and start to configure the analysis
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/1_conf_analysis.png)

First, setup the remote target access
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/2_linuxremote.png)
* 1) System profiling
Collect system performance log for 15 secs.
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/3_analyzesystem.png)
Choose **hotspot** analysis type. Click **Start button** to start collecting the performance data.
Locate where the performance logs is stored at

* 2) Launch application and profile
{TBD}
* 3) Attach to the process
{TBD}

## 6. Real-time profiling for GPU usages
    /opt/intel/vtune_amplifier/target/gpa_router &
    /opt/intel/vtune_amplifier/host/gpa-system-analyzer-bin
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/gpasystemanalyzer.png)

## 7. OpenVINO workload benchmarking
This is the prework to install OpenVINO toolkits by following the online documentation https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html

Install the dependencies

	cd /opt/intel/openvino/install_dependencies
	sudo -E ./install_openvino_dependencies.sh
	source /opt/intel/openvino/bin/setupvars.sh
	cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
	sudo ./install_prerequisites.sh
		
Run the demo

    source /opt/intel/openvino/bin/setupvars.sh
    cd /opt/intel/openvino/deployment_tools/demo
    ./demo_squeezenet_download_convert_run.sh

This verification script downloads a SqueezeNet model, uses the Model Optimizer to convert the model to the .bin and .xml Intermediate Representation (IR) files. The Inference Engine requires this model conversion so it can use the IR as input and achieve optimum performance on Intel hardware.
This verification script builds the Image Classification Sample application and run it with the car.png image located in the demo directory. When the verification script completes, you will have the label and confidence for the top-10 categories

![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/image_classification_script_output_lnx.png)

Run demo_security_barrier_camera

    ./demo_security_barrier_camera.sh
    
Follow the link to install dependencies for GPU - https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps

Follow the link to install dependencies for HDDL VPU - https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux_ivad_vpu.html

Build the openVINO samples

	/opt/intel/openvino/deployment_tools/inference_engine/samples/build_samples.sh
	ls /root/inference_engine_samples_build/intel64/Release/

Use the following commands to test security_barrier_camera sample for CPU, GPU, HDDL by using -d CPU, -d GPU, -d HDDL accordingly.

	/root/inference_engine_samples_build/intel64/Release/security_barrier_camera_demo -no-show -i /home/ieisw/Downloads/sampleVideos/car-detection.mp4 -m /root/inference_engine_samples_build/intel64/Release/dldt/vehicle-license-plate-detection-barrier-0106.xml -m_va /root/inference_engine_samples_build/intel64/Release/dldt/vehicle-attributes-recognition-barrier-0039.xml -m_lpr /root/inference_engine_samples_build/intel64/Release/dldt/license-plate-recognition-barrier-0001.xml -d CPU -d_va CPU -d_lpr CPU    
    
## 8. VTune, SoC Watch and Energy Profiler usages
Download SoC Watch for Windows package via https://registrationcenter.intel.com/en/products/
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/downloadsocwatch.png)
* collect the system performance metrics logs

	cd C:\Program Files (x86)\Intel\SoCWatch\bin\
	socwatch -f sys -t 15 -m -r vtune -o perfdata\perf1

Open the plain-text reports perf1.csv
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/socwatch_sysreport.png)

* import the system performance metrics logs perf1.pwr into VTune
Open VTune UI and click the import button
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/vtuneimport.png)

Here is the Energy Profiler usage (VTune + SoC Watch)
![image of the output](https://github.com/Joel-lin/IntelSoftwareWorkshop/blob/master/images/vtuneEnergyAnalysis.png)  
