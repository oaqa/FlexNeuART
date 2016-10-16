/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * This is a simple fix of the annoying bug:
 * not returning status zero on exiting the app.
 */

//package org.apache.uima.examples.cpe;
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;

import org.apache.uima.UIMAFramework;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionProcessingEngine;
import org.apache.uima.collection.EntityProcessStatus;
import org.apache.uima.collection.StatusCallbackListener;
import org.apache.uima.collection.metadata.CpeDescription;
import org.apache.uima.util.XMLInputSource;

/**
 * Main Class that runs a Collection Processing Engine (CPE). This class reads a CPE Descriptor as a
 * command-line argument and instantiates the CPE. It also registers a callback listener with the
 * CPE, which will print progress and statistics to System.out.
 * 
 * 
 */
public class SimpleRunCPE_fixed extends Thread {
  /**
   * The CPE instance.
   */
  private CollectionProcessingEngine mCPE;

  /**
   * Start time of CPE initialization
   */
  private long mStartTime;
  
  /**
   * Start time of the processing
   */
  private long mInitCompleteTime;

  /**
   * Constructor for the class.
   * 
   * @param args
   *          command line arguments into the program - see class description
   */
  public SimpleRunCPE_fixed(String args[]) throws Exception {
    mStartTime = System.currentTimeMillis();

    // check command line args
    if (args.length < 1) {
      printUsageMessage();
      System.exit(1);
    }

    // parse CPE descriptor
    System.out.println("Parsing CPE Descriptor");
    CpeDescription cpeDesc = UIMAFramework.getXMLParser().parseCpeDescription(
            new XMLInputSource(args[0]));
    // instantiate CPE
    System.out.println("Instantiating CPE");
    mCPE = UIMAFramework.produceCollectionProcessingEngine(cpeDesc);

    // Create and register a Status Callback Listener
    mCPE.addStatusCallbackListener(new StatusCallbackListenerImpl());

    // Start Processing
    System.out.println("Running CPE");
    mCPE.process();

    // Allow user to abort by pressing Enter
    System.out.println("To abort processing, type \"abort\" and press enter.");
    while (true) {
      String line = new BufferedReader(new InputStreamReader(System.in)).readLine();
      if ("abort".equals(line) && mCPE.isProcessing()) {
        System.out.println("Aborting...");
        mCPE.stop();
        break;
      }
    }
  }

  
  private static void printUsageMessage() {
    System.out.println(" Arguments to the program are as follows : \n"
            + "args[0] : path to CPE descriptor file");
  }

  /**
   * main class.
   * 
   * @param args
   *          Command line arguments - see class description
   */
  public static void main(String[] args) throws Exception {
    new SimpleRunCPE_fixed(args);
  }

  /**
   * Callback Listener. Receives event notifications from CPE.
   * 
   * 
   */
  class StatusCallbackListenerImpl implements StatusCallbackListener {
    int entityCount = 0;

    long size = 0;

    /**
     * Called when the initialization is completed.
     * 
     * @see org.apache.uima.collection.processing.StatusCallbackListener#initializationComplete()
     */
    public void initializationComplete() {      
      System.out.println("CPM Initialization Complete");
      mInitCompleteTime = System.currentTimeMillis();
    }

    /**
     * Called when the batchProcessing is completed.
     * 
     * @see org.apache.uima.collection.processing.StatusCallbackListener#batchProcessComplete()
     * 
     */
    public void batchProcessComplete() {
      System.out.print("Completed " + entityCount + " documents");
      if (size > 0) {
        System.out.print("; " + size + " characters");
      }
      System.out.println();
      long elapsedTime = System.currentTimeMillis() - mStartTime;
      System.out.println("Time Elapsed : " + elapsedTime + " ms ");
    }

    /**
     * Called when the collection processing is completed.
     * 
     * @see org.apache.uima.collection.processing.StatusCallbackListener#collectionProcessComplete()
     */
    public void collectionProcessComplete() {
      long time = System.currentTimeMillis();
      System.out.print("Completed " + entityCount + " documents");
      if (size > 0) {
        System.out.print("; " + size + " characters");
      }
      System.out.println();
      long initTime = mInitCompleteTime - mStartTime; 
      long processingTime = time - mInitCompleteTime;
      long elapsedTime = initTime + processingTime;
      System.out.println("Total Time Elapsed: " + elapsedTime + " ms ");
      System.out.println("Initialization Time: " + initTime + " ms");
      System.out.println("Processing Time: " + processingTime + " ms");
      
      System.out.println("\n\n ------------------ PERFORMANCE REPORT ------------------\n");
      System.out.println(mCPE.getPerformanceReport().toString());
      // stop the JVM. Otherwise main thread will still be blocked waiting for
      // user to press Enter.
      System.exit(0);
    }

    /**
     * Called when the CPM is paused.
     * 
     * @see org.apache.uima.collection.processing.StatusCallbackListener#paused()
     */
    public void paused() {
      System.out.println("Paused");
    }

    /**
     * Called when the CPM is resumed after a pause.
     * 
     * @see org.apache.uima.collection.processing.StatusCallbackListener#resumed()
     */
    public void resumed() {
      System.out.println("Resumed");
    }

    /**
     * Called when the CPM is stopped abruptly due to errors.
     * 
     * @see org.apache.uima.collection.processing.StatusCallbackListener#aborted()
     */
    public void aborted() {
      System.out.println("Aborted");
      // stop the JVM. Otherwise main thread will still be blocked waiting for
      // user to press Enter.
      System.exit(1);
    }

    /**
     * Called when the processing of a Document is completed. <br>
     * The process status can be looked at and corresponding actions taken.
     * 
     * @param aCas
     *          CAS corresponding to the completed processing
     * @param aStatus
     *          EntityProcessStatus that holds the status of all the events for aEntity
     */
    public void entityProcessComplete(CAS aCas, EntityProcessStatus aStatus) {
      if (aStatus.isException()) {
        List exceptions = aStatus.getExceptions();
        for (int i = 0; i < exceptions.size(); i++) {
          ((Throwable) exceptions.get(i)).printStackTrace();
        }
        return;
      }
      entityCount++;
      String docText = aCas.getDocumentText();
      if (docText != null) {
        size += docText.length();
      }
    }
  }

}
