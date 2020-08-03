/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.utilities;

import java.util.Random;

/**
 * 
 * @author vdang
 *
 */
public class MergeSorter {

	public static void main(String[] args)
	{
		float[][] f = new float[1000][];
		for(int r=0;r<f.length;r++)
		{
			f[r] = new float[500];
			Random rd = new Random();
			for(int i=0;i<f[r].length;i++)
			{
				//float x = rd.nextFloat();
				float x = rd.nextInt(10);
				//System.out.print(x + " ");
				f[r][i] = x;
			}
			//System.out.println("");
		}
		double start = System.nanoTime();
		for(int r=0;r<f.length;r++)
			sort(f[r], false);
		double end = System.nanoTime();		
		System.out.println("# " + (double)(end-start)/1e9 + " ");
	}
	
	public static int[] sort(float[] list, boolean asc)
	{
		return sort(list, 0, list.length-1, asc);
	}
	public static int[] sort(float[] list, int begin, int end, boolean asc)
	{
		int len = end - begin + 1;
		int[] idx = new int[len];
		int[] tmp = new int[len];
		for(int i=begin;i<=end;i++)
			idx[i-begin] = i;
		
		//identify natural runs and merge them (first iteration)
		int i=1;
		int j=0;
		int k=0;
		int start= 0;
		int[] ph = new int[len/2+3];
		ph[0] = 0;
		int p=1;
		do {
			start = i-1;
			while(i < idx.length && ((asc && list[begin+i] >= list[begin+i-1]) || (!asc && list[begin+i] <= list[begin+i-1]))) i++;
			if(i == idx.length)
			{
				System.arraycopy(idx, start, tmp, k, i-start);
				k = i;
			}
			else
			{
				j=i+1;
				while(j < idx.length && ((asc && list[begin+j] >= list[begin+j-1]) || (!asc && list[begin+j] <= list[begin+j-1]))) j++;
				merge(list, idx, start, i-1, i, j-1, tmp, k, asc);
				i = j+1;
				k=j;				
			}
			ph[p++] = k;
		}while(k < idx.length);
		System.arraycopy(tmp, 0, idx, 0, idx.length);
		
		//subsequent iterations
		while(p > 2)
		{
			if(p % 2 == 0)
				ph[p++] = idx.length;
			k=0;
			int np = 1;
			for(int w=0;w<p-1;w+=2)
			{
				merge(list, idx, ph[w], ph[w+1]-1, ph[w+1], ph[w+2]-1, tmp, k, asc);
				k = ph[w+2];
				ph[np++] = k;				
			}
			p = np;
			System.arraycopy(tmp, 0, idx, 0, idx.length);
		}		
		return idx;
	}
	private static void merge(float[] list, int[] idx, int s1, int e1, int s2, int e2, int[] tmp, int l, boolean asc)
	{
		int i=s1;
		int j=s2;
		int k=l;
		while(i <= e1 && j <= e2)
		{
			if(asc)
			{
				if(list[idx[i]] <= list[idx[j]])
					tmp[k++] = idx[i++];
				else
					tmp[k++] = idx[j++];
			}
			else
			{
				if(list[idx[i]] >= list[idx[j]])
					tmp[k++] = idx[i++];
				else
					tmp[k++] = idx[j++];
			}
		}
		while(i <= e1)
			tmp[k++] = idx[i++];
		while(j <= e2)
			tmp[k++] = idx[j++];
	}
	
	public static int[] sort(double[] list, boolean asc)
	{
		return sort(list, 0, list.length-1, asc);
	}
	public static int[] sort(double[] list, int begin, int end, boolean asc)
	{
		int len = end - begin + 1;
		int[] idx = new int[len];
		int[] tmp = new int[len];
		for(int i=begin;i<=end;i++)
			idx[i-begin] = i;
		
		//identify natural runs and merge them (first iteration)
		int i=1;
		int j=0;
		int k=0;
		int start= 0;
		int[] ph = new int[len/2+3];
		ph[0] = 0;
		int p=1;
		do {
			start = i-1;
			while(i < idx.length && ((asc && list[begin+i] >= list[begin+i-1]) || (!asc && list[begin+i] <= list[begin+i-1]))) i++;
			if(i == idx.length)
			{
				System.arraycopy(idx, start, tmp, k, i-start);
				k = i;
			}
			else
			{
				j=i+1;
				while(j < idx.length && ((asc && list[begin+j] >= list[begin+j-1]) || (!asc && list[begin+j] <= list[begin+j-1]))) j++;
				merge(list, idx, start, i-1, i, j-1, tmp, k, asc);
				i = j+1;
				k=j;				
			}
			ph[p++] = k;
		}while(k < idx.length);
		System.arraycopy(tmp, 0, idx, 0, idx.length);
		
		//subsequent iterations
		while(p > 2)
		{
			if(p % 2 == 0)
				ph[p++] = idx.length;
			k=0;
			int np = 1;
			for(int w=0;w<p-1;w+=2)
			{
				merge(list, idx, ph[w], ph[w+1]-1, ph[w+1], ph[w+2]-1, tmp, k, asc);
				k = ph[w+2];
				ph[np++] = k;				
			}
			p = np;
			System.arraycopy(tmp, 0, idx, 0, idx.length);
		}		
		return idx;
	}
	private static void merge(double[] list, int[] idx, int s1, int e1, int s2, int e2, int[] tmp, int l, boolean asc)
	{
		int i=s1;
		int j=s2;
		int k=l;
		while(i <= e1 && j <= e2)
		{
			if(asc)
			{
				if(list[idx[i]] <= list[idx[j]])
					tmp[k++] = idx[i++];
				else
					tmp[k++] = idx[j++];
			}
			else
			{
				if(list[idx[i]] >= list[idx[j]])
					tmp[k++] = idx[i++];
				else
					tmp[k++] = idx[j++];
			}
		}
		while(i <= e1)
			tmp[k++] = idx[i++];
		while(j <= e2)
			tmp[k++] = idx[j++];
	}
}
