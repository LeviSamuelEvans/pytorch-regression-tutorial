# ROOT to HDF5 Converter
This script converts data stored in ROOT files into HDF5 format with a structured layout. This is very useful for ML workflows. Some other file formats are also very popular, e.g. `parquet`, but this script is specifically designed for the HDF5 format and utilises `h5py`. It's worth doing some research yourself and experimenting with what is best for you!

---
The script is designed to be flexible, allowing you to specify the features that you want to extract from your analysis ROOT files. Furthermore, there are different classes to facilitate the processing of different types of useful particle physics data. Currently, the output structure differs from that of the `.h5` used in the tutorial, since that used an older version of the script, but this is something I will address in the future.

## Features

- **Particle-Level Processing**  
  Supports storing multiple object types (e.g., jets, electrons, muons) per event with masks handling variable object counts. This can also be applied to other particle objects, e.g. tracks. One might want to extend to this also perform mathematical operations on the objects, e.g. applying trigonometric functions to periodic variables (e.g. $\phi$).
  
- **Global Properties & Metadata**  
  Stores event-level variables (MET, jet multiplicities, etc.).
  
- **Classification & Regression Targets**  
  Optionally processes classification labels or regression targets and stores them in dedicated groups. One might want to extend to this also perform mathematical operations on the targets, e.g. ratios, log-transformations, etc.

- **Customisations**  
  Flexible to adapt the structure (max objects per event, variable mappings, etc.).

---

## Requirements  
- [uproot](https://github.com/scikit-hep/uproot4)  
- [numpy](https://numpy.org)  
- [pandas](https://pandas.pydata.org/)  
- [h5py](https://www.h5py.org/)  
- [pyyaml](https://pyyaml.org/)  
- [rich](https://github.com/Textualize/rich)  
- [tqdm](https://github.com/tqdm/tqdm)  

You can install these via:
```bash
pip install uproot numpy pandas h5py pyyaml rich tqdm
```
**Note:** you should be careful not to corrupt your global python enviroment by just running `pip install ...` without using a virtual environment. Furthermore, in reality it really is best to manage packages using conda enviroments. I'll move the setup for this tutorial to a conda enviroment soon!

## VSCode Extensions

I highly, highly recommend the following extension, if you are using VSCode:
- [h5web.vscode-h5web](https://marketplace.visualstudio.com/items?itemName=h5web.vscode-h5web)

This allows you to view the HDF5 files in the VSCode editor, which is very useful for debugging etc. It's the best tool I've found so far, and warrants the 5 stars it has!

---

## Usage

1. **Prepare a YAML file** (e.g., `variables.yaml`) specifying the features to read from the ROOT file:
   ```yaml
   features:
     - jet_pt
     - jet_eta
     - jet_phi
     - jet_e
     - el_pt
     - ...
   ```

2. **Run the script**:
   ```bash
   python convert_to_h5.py \
       -d /path/to/root/files \
       -v variables.yaml \
       -s output_store.h5 \
       -n 10000 \
       --max-jets 10
   ```
   - `-d`  : Directory containing the `.root` files.  
   - `-v`  : Path to your YAML file with variables.  
   - `-s`  : Output HDF5 filename (default is `store.h5`).  
   - `-n`  : Maximum number of events to process per file (optional: default is all).  
   - `--max-jets` : Maximum number of jets to store for each event (optional: default is 10).  
   - `-O/--overwrite` : Overwrites the HDF5 file if it already exists (default is on so be careful! We have included a user prompt to ask you if you are sure you want to overwrite the file, so maybe not best for any batch stuff.).

3. **Check the output**  
   The resulting `output_store.h5` file will contain structured groups (e.g., `Files/filename_INPUTS/JETS`, `TARGETS/CLASSIFICATION`, etc.) with your data. An example is shown below.

---
## Output Structure example

Here is an example of an output file structure after running the script:

```bash
|-Files                         
|---combined_ttH_PP8_mc16a_AFII_root
|-----INPUTS                    
|-------ELECTRONS               
|---------MASK                  
|---------charge                
|---------energy                
|---------eta                   
|---------phi                   
|---------pt                    
|-------GLOBAL_PROPERTIES       
|---------HT_all                
|---------met_met               
|---------met_phi               
|---------nJets                 
|-------JETS                    
|---------MASK                  
|---------btag                  
|---------energy                
|---------eta                   
|---------phi                   
|---------pt                    
|-------METADATA                
|---------weight_jvt            
|---------weight_leptonSF       
|---------weight_mc             
|---------weight_normalise      
|---------weight_pileup         
|-------MUONS                   
|---------MASK                  
|---------charge                
|---------energy                
|---------eta                   
|---------phi                   
|---------pt                    
|-----TARGETS                   
|-------CLASSIFICATION          
|---------event_class           
|-------REGRESSION              
|---------higgs_pt              
|---combined_ttH_PP8_mc16d_AFII_root
```
I have included this `test.h5` file in the repo in case you want to inspect it using the VSCode extension mentioned above.

## Notes & Customisation

- **Particle Processors**: Modify or add `ParticleProcessor` instances to handle different object types (e.g., photons, secondary vertices).  
- **Metadata**: Extend or modify `MetaDataProcessor` to store additional weights or other event-level metadata you may need.  
- **Classification/Regression**: Adjust class maps, add more processors, or change variable mappings to fit your specific classification/regression tasks.
- **Parallelisation**: For large data sets, you could adapt the script to process files in parallel or in batches. To be honest, it is already pretty fast! :D

---

**Happy Converting!** If you have any issues or feature requests, feel free to adapt or extend the script to meet your analysis needs.

Run `python convert_to_h5.py -h` to see the help message and usage instructions. (check .assets/help_message.png for reference example.)