"""Example usage of ProxAI to simulate Dungeon and Dragons master.

Example response from OpenAI's GPT-3.5-turbo model:
A Dungeons & Dragons (D&D) Dungeon Master (DM) should ideally possess a
variety of qualities and skills to effectively run a D&D campaign. Some
important properties that a DM should have include:
1. Creativity: Being able to create imaginative and engaging storylines,
characters, and encounters is essential for keeping players interested and
immersed in the game.
2. Flexibility: Being able to adapt and think on their feet when players go
off course or unexpected situations arise during the game.
3. Organization: Keeping track of various storylines, characters, and game
mechanics requires good organizational skills.
4. Knowledge of the rules: Understanding the rules of D&D and being able to
interpret and apply them fairly and consistently is crucial for a DM.
5. Patience: Being patient with players, especially new ones who may need
guidance or clarification on rules and mechanics.
6. Communication skills: Being able to effectively communicate with players,
both in terms of conveying the story and rules, as well as soliciting
feedback and resolving conflicts.
7. Leadership: Guiding the players through the game and making decisions
that keep the game flowing smoothly.
8. Empathy: Understanding and considering the needs and preferences of the
players to create an enjoyable gaming experience for everyone.
Overall, a good DM should be a storyteller, a referee, and a collaborator,
working with the players to create memorable and enjoyable adventures in the
world of D&D.

Example response from Claude's claude-3-opus-20240229 model:
A good Dungeon Master (DM) in Dungeons and Dragons should possess several key
qualities:
1. Creativity: A DM should be able to create engaging storylines, interesting
non-player characters (NPCs), and immersive environments for the players to
explore.
2. Adaptability: No matter how well-planned a campaign is, players often make
unexpected decisions. A good DM should be able to think on their feet and adapt
the story accordingly.
3. Knowledge of the rules: While the rules can be flexible, a DM should have a
solid grasp of the game's mechanics to ensure fair and consistent gameplay.
4. Impartiality: A DM should be fair and impartial, not favoring any particular
player or their own NPCs over the others.
5. Good communication skills: A DM should be able to effectively describe
scenes, convey NPC dialogue, and communicate with players to ensure everyone
is on the same page.
6. Organizational skills: Running a campaign involves keeping track of various
storylines, characters, and game elements. Good organizational skills are
essential for a smooth game experience.
7. Enthusiasm: A passionate and enthusiastic DM can make the game more
enjoyable for everyone involved, keeping players engaged and excited about
the unfolding story.
8. Empathy: A good DM should be able to understand and cater to the needs and
preferences of their players, ensuring that everyone is comfortable and
enjoying the game.
"""

import proxai


def brainstorm_dd_master_properties():
  text = proxai.generate_text(
      'What kind of properties that dungeon and dragons master needs to have?')
  print(text)


def main():
  proxai.register_model('openai', 'gpt-3.5-turbo')
  brainstorm_dd_master_properties()

  proxai.register_model('claude', 'claude-3-opus-20240229')
  brainstorm_dd_master_properties()


if __name__ == '__main__':
  main()
