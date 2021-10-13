package ciir.umass.edu.features;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.Map;
import java.util.TreeMap;
import java.util.Set;
import java.util.Iterator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

    
/*
 * Calculate feature use statistics on saved model files that make use of only a subset
 * of all defined training features.  This may be useful for persons attempting to restrict
 * a feature set by possibly eliminating features that are dropped or rarely used by
 * ranking resulting models.  Experimentation is still required to confirm absent or rarely
 * used features have little or no effect on model effectiveness.
 */

public class FeatureStats {

  private String modelName;
  private String modelFileName;
  private File f;
  private BufferedReader br;


  /**
   * Define the saved model file to be used.
   *
   * @param   model file name
   */
  protected FeatureStats (String modelFileName) {

    try {
      this.f = new File (modelFileName);
      this.modelFileName = f.getAbsolutePath ();
      this.br = new BufferedReader (new FileReader (f));

      //- Remove leading ## from model name and handle multiple word names
      String modelLine = br.readLine().trim();
      String[] nameparts = modelLine.split(" ");
      int len = nameparts.length;
    
      if (len == 2) {
        this.modelName = nameparts[1].trim();
      }
      else if (len == 3) {
        this.modelName = nameparts[1].trim() + " " + nameparts[2].trim();
      }
    }
    catch (IOException ioex) {
	System.out.println ("IOException opening model file " + modelFileName + ". Quitting.");
	System.exit (1);
    }
  }  //- end constructor

    
  private TreeMap<Integer, Integer> getFeatureWeightFeatureFrequencies () {

    TreeMap<Integer, Integer>tm = new TreeMap<>();
    
    try {
      String line = null;
      while ( (line = this.br.readLine ()) != null) {
        line = line.trim ().toLowerCase ();

        if (line.length () == 0) {
          continue;
        }

	//- Not interested in model comments
        else if (line.contains ("##")) {
          continue;
        }
	
        //- A RankBoost model contains one line with all the feature weights so
	//  we need to split the line up into an array of feature and their weights.
	else {
          String[] featureLines = line.split (" ");
          int featureFreq = 0;
	  
	  for (int i=0; i<featureLines.length; i++) {
            Integer featureID = Integer.valueOf (featureLines[i].split(":")[0]);

            if (tm.containsKey (featureID)) {
              featureFreq = tm.get (featureID);
              featureFreq++;
              tm.put (featureID, featureFreq);
            }
            else {
              tm.put (featureID, 1);
            }
	  }
	}
      }  //- end while reading

      //br.close ();
    }  //- end try
    catch (Exception ex) {
      System.out.println ("Exception: " + ex.toString ());
      System.exit (1);
    }

    return tm;
  }  //- end method getFeatureWeightFeatureFrequencies
    

  private TreeMap<Integer, Integer> getTreeFeatureFrequencies () {

    TreeMap<Integer, Integer> tm = new TreeMap<>();
    
    try {
      String line = null;
      while ( (line = br.readLine ()) != null) {
        line = line.trim ().toLowerCase ();

        if (line.length () == 0) {
          continue;
        }

	//- Ignore model comments
	else if (line.contains ("##")) {
          continue;
	}

        //- Generate feature frequencies
        else if (line.contains ("<feature>")) {
          int quote1 = line.indexOf ('>', 0);
          int quote2 = line. indexOf ('<', quote1+1);
          String featureIdStr = line.substring (quote1+1, quote2);
          Integer featureID = Integer.valueOf (featureIdStr.trim ());

          if (tm.containsKey (featureID)) {
            int featureFreq = tm.get (featureID);
            featureFreq++;
            tm.put (featureID, featureFreq);
          }
          else {
            tm.put (featureID, 1);
          }
        }
      }  //- end while reading
    }  //- end try
    catch (Exception ex) {
      System.out.println ("Exception: " + ex.toString ());
      System.exit (1);
    }

    return tm;
      
  }  //- end method getTreeFeatureFrequencies

    
  public void writeFeatureStats () {
    int featureMin   = Integer.MAX_VALUE;
    int featureMax   = 0;
    int featuresUsed = 0;
    int featureFreq  = 0;
    String modelName = this.modelName;
    TreeMap<Integer, Integer>featureTM = null;
    
    try {

      //- There should be a model name in the file or something is screwy.
      if (modelName == null) {
        System.out.println ("No model name defined.  Quitting.");
        System.exit (1);
      }

      //- Can't do feature statistics on models that make use of every feature as it is
      //  then difficult to say the statistics mean anything.
      if ( modelName.equals ("Coordinate Ascent") || 
           modelName.equals ("LambdaRank") ||
           modelName.equals ("Linear Regression") ||
           modelName.equals ("ListNet") || 
           modelName.equals ("RankNet") ) {
        System.out.println (modelName + " uses all features.  Can't do selected model statistics for this algorithm.");
        System.exit (0);
      }

      //- Feature:Weight models
      else if ( modelName.equals ("AdaRank") ||
                modelName.equals ("RankBoost")) {
             featureTM = getFeatureWeightFeatureFrequencies ();
      }

      //- Tree models
      else if ( modelName.equals ("LambdaMART") ||
                modelName.equals ("MART") ||
                modelName.equals ("Random Forests") ) {
        featureTM = getTreeFeatureFrequencies ();
      }

      br.close ();
    }
    catch (IOException ioe) {
      System.out.println ("IOException on file " + modelFileName);
      System.exit (1);
    }

    //- How many features?
    featuresUsed = featureTM.size ();

    //- Print the feature frequencies and statistics
    System.out.println ("\nModel File: " + modelFileName);
    System.out.println ("Algorithm : " + modelName);
    System.out.println ("");
    System.out.println ("Feature frequencies : ");
    
    Set s = featureTM.entrySet ();
    DescriptiveStatistics ds = new DescriptiveStatistics ();

    Iterator it = s.iterator ();
    while (it.hasNext ()) {
      Map.Entry e = (Map.Entry)it.next ();
      int freqID = (int)e.getKey ();
      int freq = (int)e.getValue ();
      System.out.printf ("\tFeature[%d] : %7d\n", freqID, freq);
      ds.addValue (freq);	
    }

    //- Print out summary statistics
    System.out.println (" ");
    System.out.printf ("Total Features Used: %d\n\n", featuresUsed);
    System.out.printf ("Min frequency    : %10.2f\n", ds.getMin());
    System.out.printf ("Max frequency    : %10.2f\n", ds.getMax());
    //System.out.printf ("Q1    : %10.2f\n", ds.getPercentile (25));
    System.out.printf ("Median frequency : %10.2f\n", ds.getPercentile (50));
    //System.out.printf ("Q3    : %10.2f\n", ds.getPercentile (75));
    System.out.printf ("Avg frequency    : %10.2f\n", ds.getMean ());
    System.out.printf ("Variance         : %10.2f\n", ds.getVariance ());
    System.out.printf ("STD              : %10.2f\n", ds.getStandardDeviation ());
  }  //- end writeFeatureStats

}  //- end class FeatureStats
