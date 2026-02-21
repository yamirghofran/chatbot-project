import { useState } from "react";
import { Pencil } from "lucide-react";
import type { Book, User, List, RatedBook, ActivityItem } from "@/lib/types";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { ProfileHeader } from "./ProfileHeader";
import { FavoriteBooks } from "./FavoriteBooks";
import { RatingsDistribution } from "./RatingsDistribution";
import { RatedBookList } from "./RatedBookList";
import { ActivityFeed } from "@/components/discovery/ActivityFeed";
import { ListRow } from "@/components/list/ListRow";

export type UserPageProps = {
  user: User;
  isOwnProfile?: boolean;
  favorites: Book[];
  ratedBooks: RatedBook[];
  activity: ActivityItem[];
  lists: List[];
  followingCount?: number;
  followersCount?: number;
  onFollow?: () => void;
  onEditProfile?: () => void;
  onViewAllBooks?: () => void;
  onCreateList?: () => void;
  onLogout?: () => void;
};

export function UserPage({
  user,
  isOwnProfile,
  favorites,
  ratedBooks,
  activity,
  lists,
  followingCount,
  followersCount,
  onFollow,
  onEditProfile,
  onViewAllBooks,
  onCreateList,
  onLogout,
}: UserPageProps) {
  const [isEditingFavorites, setIsEditingFavorites] = useState(false);

  return (
    <div>
      <ProfileHeader
        user={user}
        isOwnProfile={isOwnProfile}
        followingCount={followingCount}
        followersCount={followersCount}
        onFollow={onFollow}
        onEditProfile={onEditProfile}
        onLogout={onLogout}
      />

      <Separator className="my-6" />

      <div className="grid grid-cols-2 gap-8">
        <section>
          <div className="flex items-center gap-2 mb-2">
            <h2 className="font-heading text-lg font-semibold">Top 3 Favorites</h2>
            {isOwnProfile && (
              <Button
                variant={isEditingFavorites ? "secondary" : "ghost"}
                size="icon-sm"
                onClick={() => setIsEditingFavorites((prev) => !prev)}
                aria-label="Edit favorites"
              >
                <Pencil className="size-3.5" />
              </Button>
            )}
          </div>
          <FavoriteBooks
            books={favorites}
            username={user.handle}
            isOwnProfile={isOwnProfile}
            isEditing={isEditingFavorites}
          />
        </section>
        <section>
          <h2 className="font-heading text-lg font-semibold mb-2">Ratings</h2>
          <RatingsDistribution ratedBooks={ratedBooks} />
        </section>
      </div>

      <Separator className="my-6" />

      <section>
        <h2 className="font-heading text-lg font-semibold mb-2">Library</h2>
        <RatedBookList ratedBooks={ratedBooks} onViewAll={onViewAllBooks} />
      </section>

      <Separator className="my-6" />

      <section>
        <h2 className="font-heading text-lg font-semibold mb-2">Recent Activity</h2>
        {activity.length > 0 ? (
          <ActivityFeed items={activity.slice(0, 5)} />
        ) : (
          <p className="text-sm text-muted-foreground py-4">No recent activity.</p>
        )}
      </section>

      <Separator className="my-6" />

      <section>
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="font-heading text-lg font-semibold">Lists</h2>
          {isOwnProfile && (
            <Button variant="outline" size="sm" onClick={onCreateList}>
              New list
            </Button>
          )}
        </div>
        {lists.length > 0 ? (
          <div>
            {lists.map((list, i) => (
              <div key={list.id}>
                {i > 0 && <Separator />}
                <ListRow list={list} />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground py-4">No lists yet.</p>
        )}
      </section>
    </div>
  );
}
