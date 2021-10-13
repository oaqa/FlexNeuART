/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.utilities;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class ExpressionEvaluator {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		ExpressionEvaluator ev = new ExpressionEvaluator();
		String exp = "sqrt(16)/exp(4^2)";
		System.out.println(ev.getRPN(exp) + "");
		System.out.println(ev.eval(exp) + "");
	}

	class Queue {
		private List<String> l = new ArrayList<String>();
		public void enqueue(String s)
		{
			l.add(s);			
		}
		public String dequeue()
		{
			if(l.size() == 0)
				return "";
			String s = l.get(0);
			l.remove(0);
			return s;
		}
		public int size()
		{
			return l.size();
		}
		public String toString()
		{
			String output = "";
			for(int i=0;i<l.size();i++)
				output += l.get(i) + " ";
			return output.trim();
		}
	}
	class Stack {
		private List<String> l = new ArrayList<String>();
		public void push(String s)
		{
			l.add(s);			
		}
		public String pop()
		{
			if(l.size() == 0)
				return "";
			String s = l.get(l.size()-1);
			l.remove(l.size()-1);
			return s;
		}
		public int size()
		{
			return l.size();
		}
		public String toString()
		{
			String output = "";
			for(int i=l.size()-1;i>=0;i--)
				output += l.get(i) + " ";
			return output.trim();
		}
	}
	
	private static String[] operators = new String[]{"+", "-", "*", "/", "^"};
	private static String[] functions = new String[]{"log", "ln", "log2", "exp", "sqrt", "neg"};
	private static HashMap<String, Integer> priority = null;
	
	private boolean isOperator(String token)
	{
		for(int i=0;i<operators.length;i++)
			if(token.compareTo(operators[i]) == 0)
				return true;
		return false;
	}
	private boolean isFunction(String token)
	{
		for(int i=0;i<functions.length;i++)
			if(token.compareTo(functions[i]) == 0)
				return true;
		return false;
	}
	private Queue toPostFix(String expression)
	{
		expression = expression.replace(" ", "");
		Queue output = new Queue();
		Stack op = new Stack();
		String lastReadToken = "";
		for(int i=0;i<expression.length();i++)
		{
			String token = expression.charAt(i) + "";
			if(token.compareTo("(") == 0)
				op.push(token);
			else if(token.compareTo(")") == 0)
			{
				boolean foundOpen = false;
				while(op.size() > 0 && !foundOpen)
				{					
					String last = op.pop();
					if(last.compareTo("(") != 0)
						output.enqueue(last);
					else
						foundOpen = true;
				}
				if(!foundOpen)
				{
					throw RankLibError.create("Error: Invalid expression: \"" + expression + "\". Parentheses mismatched.");
				}
			}
			else if(isOperator(token))
			{
				if(lastReadToken.compareTo("(") == 0 || isOperator(lastReadToken))//@token is a unary opeartor (e.g. +2, -3)
				{
					if(token.compareTo("-") == 0)//convert it to a neg function
						op.push("neg");
					//if it's a unary "+", we can just ignore it
				}
				else
				{
					if(op.size() > 0)
					{
						String last = op.pop();
						if(last.compareTo("(") == 0)
							op.push(last);//push the "(" back in
						else if(priority.get(token) > priority.get(last))
							op.push(last);//push the last operator back into the stack 
						else if(priority.get(token) < priority.get(last))
							output.enqueue(last);
						else //equal priority
						{
							if(token.compareTo("^") == 0)//right-associative operator
								op.push(last);//push the last operator back into the stack
							else
								output.enqueue(last);
						}
					}
					op.push(token);
				}
			}
			else //maybe function, maybe operand
			{
				int j=i+1;
				while(j < expression.length())
				{
					String next = expression.charAt(j) + "";
					if(next.compareTo(")")==0 || next.compareTo("(")==0 || isOperator(next))
						break;
					else
					{
						token += next;
						j++;
					}
				}
				i = j-1;
				//test again to see if @token now matches a function or not
				if(isFunction(token))
				{
					if(j == expression.length())
					{
						throw RankLibError.create("Error: Invalid expression: \"" + expression + "\". Function specification requires parentheses.");
					}
					if(expression.charAt(j) != '(')
					{
						throw RankLibError.create("Error: Invalid expression: \"" + expression + "\". Function specification requires parentheses.");
					}
					op.push(token);
				}
				else//operand
				{
					try {
						Double.parseDouble(token);						
					}
					catch(Exception ex)
					{
						throw RankLibError.create("Error: \"" + token + "\" is not a valid token.");
					}
					output.enqueue(token);
				}
			}
			lastReadToken = token;
		}
		while(op.size() > 0)
		{
			String last = op.pop();
			if(last.compareTo("(") == 0)
			{
				throw RankLibError.create("Error: Invalid expression: \"" + expression + "\". Parentheses mismatched.");
			}
			output.enqueue(last);
		}
		return output;
	}
	
	public ExpressionEvaluator()
	{
		if(priority == null)
		{
			priority = new HashMap<String, Integer>();
			priority.put("+", 2);
			priority.put("-", 2);
			priority.put("*", 3);
			priority.put("/", 3);
			priority.put("^", 4);
			priority.put("neg", 5);
			priority.put("log", 6);
			priority.put("ln", 6);
			priority.put("sqrt", 6);
		}
	}
	public String getRPN(String expression)
	{
		return toPostFix(expression).toString();
	}
	public double eval(String expression)
	{
		Queue output = toPostFix(expression);
		double[] eval = new double[output.size()];
		int cp = 0;//current position
		try {
			while(output.size() > 0)
			{
				String token = output.dequeue();
				double v = 0;
				if(isOperator(token))
				{
					if(token.compareTo("+") == 0)
						v = eval[cp-2] + eval[cp-1];
					else if(token.compareTo("-") == 0)
						v = eval[cp-2] + eval[cp-1];
					else if(token.compareTo("*") == 0)
						v = eval[cp-2] * eval[cp-1];
					else if(token.compareTo("/") == 0)
						v = eval[cp-2] / eval[cp-1];
					else if(token.compareTo("^") == 0)
						v = Math.pow(eval[cp-2], eval[cp-1]);
					eval[cp-2] = v;
					cp--;
				}
				else if(isFunction(token))
				{
					if(token.compareTo("log") == 0)
					{
						if(eval[cp-1] < 0)
						{
							throw RankLibError.create("Error: expression " + expression + " involves taking log of a non-positive number");
						}
						v = Math.log10(eval[cp-1]);
					}
					else if(token.compareTo("ln") == 0)
					{
						if(eval[cp-1] < 0)
						{
							throw RankLibError.create("Error: expression " + expression + " involves taking log of a non-positive number");
						}
						v = Math.log(eval[cp-1]);
					}
					else if(token.compareTo("log2") == 0)
					{
						if(eval[cp-1] < 0)
						{
							throw RankLibError.create("Error: expression " + expression + " involves taking log of a non-positive number");
						}
						v = Math.log(eval[cp-1])/Math.log(2);
					}
					else if(token.compareTo("exp") == 0)
						v = Math.exp(eval[cp-1]);
					else if(token.compareTo("sqrt") == 0)
					{
						if(eval[cp-1] < 0)
						{
							throw RankLibError.create("Error: expression " + expression + " involves taking square root of a negative number");
						}
						v = Math.sqrt(eval[cp-1]);
					}
					else if(token.compareTo("neg") == 0)
						v = - eval[cp-1];
					eval[cp-1] = v;
				}
				else//operand
					eval[cp++] = Double.parseDouble(token);
			}
			if(cp != 1)
			{
				throw RankLibError.create("Error: invalid expression: " + expression);
			}
		}
		catch(Exception ex)
		{
			throw RankLibError.create("Unknown error in ExpressionEvaluator::eval() with \"" + expression + "\"", ex);
		}
		return eval[cp-1];
	}
}
